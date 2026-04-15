"""
tools/browser.py
----------------
Lightweight web grounding tool for the self-correcting pipeline.

Two public functions
--------------------
fetch_page(url, max_chars)
    Fetches a single URL using the w3m CLI browser (subprocess) and returns
    plain text truncated to max_chars.  Falls back to a simple urllib GET
    when w3m is not installed (Windows-safe).

search_and_browse(query, num_urls, max_chars_per_page)
    Uses DuckDuckGo (duckduckgo-search) to find the top URLs for a query,
    then fetches up to num_urls of them with fetch_page.  Returns a list of
    {"url": ..., "content": ...} dicts.

The functions are intentionally simple and defensive:
- Every network / subprocess call is wrapped in try/except.
- Failures return empty strings rather than raising.
- Content is always truncated to avoid flooding the refiner prompt.

Dependencies
------------
    pip install duckduckgo-search

w3m is optional but recommended for cleaner text extraction:
    - Linux/Mac : apt install w3m  /  brew install w3m
    - Windows   : not available; urllib fallback is used automatically.
"""

from __future__ import annotations

import subprocess
import urllib.request
import urllib.error
from typing import List, Dict, Optional

from utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_W3M_TIMEOUT   = 10          # seconds per subprocess call
_URLLIB_TIMEOUT = 10          # seconds per urllib call
_DEFAULT_MAX_CHARS = 1800     # ~450 tokens — enough context, not overwhelming
_DEFAULT_NUM_URLS  = 2        # browse at most this many URLs per query

_URLLIB_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; SelfCorrectingLLM/1.0; "
        "+https://github.com/self-improving-llm)"
    )
}


# ---------------------------------------------------------------------------
# Single-page fetcher
# ---------------------------------------------------------------------------

def fetch_page(url: str, max_chars: int = _DEFAULT_MAX_CHARS) -> str:
    """
    Return plain-text content of *url*, truncated to *max_chars*.

    Tries w3m first (cleaner text), falls back to urllib (always available).
    Returns an empty string on any error — never raises.

    Parameters
    ----------
    url:
        The web page to fetch.
    max_chars:
        Hard cap on returned characters.  Content beyond this is silently
        dropped.  A value of 1800 gives roughly 400–450 tokens.
    """
    text = _fetch_w3m(url) or _fetch_urllib(url)
    if not text:
        logger.warning("browser: could not retrieve %s", url)
        return ""

    text = text.strip()
    if len(text) > max_chars:
        text = text[:max_chars] + "\n[... truncated ...]"

    logger.info("browser: fetched %d chars from %s", len(text), url)
    return text


def _fetch_w3m(url: str) -> Optional[str]:
    """Run ``w3m -dump <url>`` and return stdout, or None on failure."""
    try:
        result = subprocess.run(
            ["w3m", "-dump", url],
            capture_output=True,
            text=True,
            timeout=_W3M_TIMEOUT,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout
    except FileNotFoundError:
        pass  # w3m not installed — silent fallback
    except subprocess.TimeoutExpired:
        logger.warning("browser: w3m timed out on %s", url)
    except Exception as exc:
        logger.warning("browser: w3m error on %s: %s", url, exc)
    return None


def _fetch_urllib(url: str) -> Optional[str]:
    """Fetch *url* with urllib and return decoded text, or None on failure."""
    try:
        req = urllib.request.Request(url, headers=_URLLIB_HEADERS)
        with urllib.request.urlopen(req, timeout=_URLLIB_TIMEOUT) as resp:
            raw = resp.read()
            charset = resp.headers.get_content_charset("utf-8") or "utf-8"
            html = raw.decode(charset, errors="replace")
            return _strip_html(html)
    except urllib.error.URLError as exc:
        logger.warning("browser: urllib error on %s: %s", url, exc)
    except Exception as exc:
        logger.warning("browser: unexpected error on %s: %s", url, exc)
    return None


def _strip_html(html: str) -> str:
    """
    Very light HTML stripping — removes tags and collapses whitespace.
    Not a full parser; good enough for extracting readable prose.
    """
    import re
    # Remove <script> and <style> blocks entirely
    html = re.sub(r"<(script|style)[^>]*>.*?</(script|style)>", " ", html,
                  flags=re.DOTALL | re.IGNORECASE)
    # Strip all remaining tags
    html = re.sub(r"<[^>]+>", " ", html)
    # Decode common HTML entities
    html = (html
            .replace("&amp;",  "&")
            .replace("&lt;",   "<")
            .replace("&gt;",   ">")
            .replace("&quot;", '"')
            .replace("&#39;",  "'")
            .replace("&nbsp;", " "))
    # Collapse whitespace
    html = re.sub(r"\s+", " ", html)
    return html.strip()


# ---------------------------------------------------------------------------
# Search + browse pipeline
# ---------------------------------------------------------------------------

def search_and_browse(
    query: str,
    num_urls: int = _DEFAULT_NUM_URLS,
    max_chars_per_page: int = _DEFAULT_MAX_CHARS,
) -> List[Dict[str, str]]:
    """
    Search DuckDuckGo for *query* and browse the top results.

    Parameters
    ----------
    query:
        The search query (usually the same as the pipeline query).
    num_urls:
        Maximum number of URLs to browse.  Capped at 3 regardless.
    max_chars_per_page:
        Character limit per page passed to ``fetch_page``.

    Returns
    -------
    list of {"url": str, "content": str}
        One entry per successfully fetched page.  Empty list if search
        fails or all fetches fail.

    Notes
    -----
    Requires ``pip install duckduckgo-search``.  Fails gracefully (returns
    []) when the package is missing or the search times out.
    """
    num_urls = min(num_urls, 3)  # hard cap — never browse more than 3
    urls = _ddg_search(query, max_results=num_urls + 1)  # +1 buffer for failures

    results: List[Dict[str, str]] = []
    for url in urls[:num_urls]:
        content = fetch_page(url, max_chars=max_chars_per_page)
        if content:
            results.append({"url": url, "content": content})
        if len(results) >= num_urls:
            break

    logger.info(
        "search_and_browse: query=%r  urls_tried=%d  pages_retrieved=%d",
        query[:60], len(urls), len(results),
    )
    return results


def _ddg_search(query: str, max_results: int = 3) -> List[str]:
    """
    Return up to *max_results* URLs from DuckDuckGo for *query*.
    Returns [] if duckduckgo-search is not installed or the search fails.
    """
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        logger.warning(
            "browser: duckduckgo-search not installed. "
            "Run: pip install duckduckgo-search"
        )
        return []

    try:
        with DDGS() as ddgs:
            hits = ddgs.text(query, max_results=max_results)
            return [h["href"] for h in hits if "href" in h]
    except Exception as exc:
        logger.warning("browser: DDG search failed: %s", exc)
        return []
