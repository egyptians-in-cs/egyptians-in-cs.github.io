"""Google Scholar lookups, used sparingly.

Scholar has no API and blocks anything that looks like bulk scraping — a pass
over the whole directory fails on more than half of it. So the pipeline only
ever asks about a handful of researchers at a time:

  * the h-index gate on a new submission (a few lookups per fetch)
  * `refresh`, which takes the N stalest entries and stops at the first sign
    of being throttled

Nothing here retries aggressively; being blocked is a reason to come back
later, not to push harder.
"""

import os
import re
import sys
import time
import atexit
import signal
import random
import threading
import subprocess
from contextlib import contextmanager

SCHOLAR_ID = re.compile(r"user=([\w-]+)")

# Scholar tolerates a trickle. These gaps are deliberately generous.
MIN_DELAY = 3.0
MAX_DELAY = 8.0


class Blocked(Exception):
    """Scholar refused the request (rate limited or captcha-walled)."""


class Timeout(BaseException):
    """A lookup outran its wall-clock budget.

    Deliberately derived from BaseException rather than Exception: scholarly
    catches Exception broadly in its own retry loops, and would swallow this
    before it ever reached us.
    """


def scholar_id(profile_url):
    """The user= id out of a Scholar profile link, or None."""
    match = SCHOLAR_ID.search(profile_url or "")
    return match.group(1) if match else None


def pause():
    time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))


# scholarly.fill() downloads the researcher's entire publication list by default,
# which is far more than the directory needs (and far more requests).
SECTIONS = ["basics", "indices"]

# Left alone, scholarly retries a blocked request for many minutes before it
# raises, which hides the block from the caller's circuit breaker. Fail fast
# instead: being turned away is a signal to stop, not to keep knocking.
RETRIES = 2
TIMEOUT = 20

# set_timeout above bounds a single HTTP request, but not scholarly's own
# behaviour around it: when it suspects throttling it sleeps 60-120s internally
# between retries, and a lookup has been observed to sit there for 17 minutes.
# That hides the stall from the caller's circuit breaker, which only counts
# refusals. This is the hard ceiling on one lookup, start to finish.
MAX_LOOKUP_SECONDS = 90

_configured = False


def _configure(scholarly):
    global _configured
    if not _configured:
        scholarly.set_retries(RETRIES)
        scholarly.set_timeout(TIMEOUT)
        atexit.register(close_browser)
        _configured = True


def _descendants(pid):
    """Every process started, directly or indirectly, by `pid`."""
    try:
        lister = subprocess.Popen(["ps", "-eo", "pid=,ppid="], stdout=subprocess.PIPE, text=True)
        listing, _ = lister.communicate(timeout=10)
    except Exception:
        return []

    children = {}
    for line in listing.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            if int(parts[0]) != lister.pid:     # ps lists itself; it is not a leftover
                children.setdefault(int(parts[1]), []).append(int(parts[0]))

    found, frontier = [], [pid]
    while frontier:
        for child in children.get(frontier.pop(), []):
            found.append(child)
            frontier.append(child)
    return found


def close_browser():
    """Shut down the browser scholarly opens when Scholar shows a CAPTCHA.

    scholarly starts a geckodriver and a headless Firefox to look at CAPTCHA
    pages, and only closes them from a destructor that runs during interpreter
    shutdown - by which point it can no longer import what it needs ("Could not
    close webdriver cleanly"). Left alone, every run that meets a CAPTCHA leaves
    a driver and four Firefox processes behind.

    Quit the driver properly first; then, as a backstop, stop anything this
    process started that is still running. Only our own descendants are ever
    touched, so a Firefox you have open yourself is never affected. Returns
    how many leftover processes had to be stopped.
    """
    client = sys.modules.get("scholarly")       # never import it just to clean up
    nav = getattr(getattr(client, "scholarly", None), "_Scholarly__nav", None)
    for manager in (getattr(nav, "pm1", None), getattr(nav, "pm2", None)):
        driver = getattr(manager, "_webdriver", None)
        if driver is not None:
            try:
                driver.quit()
            except Exception:
                pass
            manager._webdriver = None            # so scholarly's destructor skips it

    leftovers = _descendants(os.getpid())
    for pid in leftovers:
        try:
            os.kill(pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
    return len(leftovers)


@contextmanager
def time_limit(seconds):
    """Abort the enclosed block if it runs longer than `seconds`.

    Uses SIGALRM, which interrupts the call wherever it is — including inside
    scholarly's internal sleeps. Only the main thread can carry an alarm, so
    off the main thread (or on a platform without SIGALRM) this does nothing
    rather than failing.
    """
    usable = hasattr(signal, "SIGALRM") and threading.current_thread() is threading.main_thread()
    if not usable:
        yield
        return

    def ring(signum, frame):
        raise Timeout(f"no answer within {seconds}s")

    previous = signal.signal(signal.SIGALRM, ring)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def fetch_author(author_id):
    """Read one Scholar profile. Raises Blocked when Scholar turns us away.

    A lookup that stalls past MAX_LOOKUP_SECONDS is reported as Blocked too:
    in practice a stall means the same thing as a refusal, and it should count
    towards the caller's circuit breaker the same way.
    """
    from scholarly import scholarly

    _configure(scholarly)
    try:
        with time_limit(MAX_LOOKUP_SECONDS):
            return scholarly.fill(scholarly.search_author_id(author_id), sections=SECTIONS)
    except Timeout as error:
        raise Blocked(f"stalled, gave up after {MAX_LOOKUP_SECONDS}s ({error})") from error
    except Exception as error:
        message = str(error).lower()
        if "cannot fetch" in message or "429" in message or "captcha" in message \
                or "maxtries" in message or "blocked" in message:
            raise Blocked(str(error)) from error
        raise


def metrics(author):
    """The fields the directory keeps from a Scholar profile."""
    return {
        "hindex": author.get("hindex", -1),
        "citedby": author.get("citedby", 0),
        "affiliation": author.get("affiliation", ""),
        "website": author.get("homepage", ""),
        "interests": author.get("interests", []) or [],
        "photo_url": author.get("url_picture", ""),
    }


def metrics_for(profile_url):
    """Metrics behind a Scholar profile link. Returns None when there is no id.

    The whole record is fetched either way, so callers take the citation count
    with the h-index rather than discarding it and leaving `citedby` at 0.
    """
    author_id = scholar_id(profile_url)
    if not author_id:
        return None
    return metrics(fetch_author(author_id))
