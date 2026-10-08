"""Focused tools for common GAIA question types.

Add or improve capabilities here. Keep each tool small, typed, documented, and
easy to test; an agent is only as reliable as the observations its tools return.
"""

from __future__ import annotations

import ast
import math
import operator
import os
import re
import socket
import subprocess
import sys
import tempfile
import threading
from contextvars import ContextVar
from io import BytesIO
from ipaddress import ip_address
from pathlib import Path
from urllib.parse import parse_qs, unquote, urljoin, urlparse
from uuid import uuid4

import pandas as pd
import requests
from bs4 import BeautifulSoup
from ddgs import DDGS
from huggingface_hub import InferenceClient
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field

MAX_TOOL_TEXT = 14_000
HTTP_TIMEOUT = 20
MAX_PDF_BYTES = 20_000_000
MAX_PDF_PAGES = 80
_SEARCH_SESSION: ContextVar[str] = ContextVar("gaia_search_session", default="default")
_TASK_QUESTION: ContextVar[str] = ContextVar("gaia_task_question", default="")
_SEARCH_HISTORY: dict[str, set[str]] = {}
_SEARCH_HISTORY_LOCK = threading.Lock()
_AGGREGATOR_HOSTS = (
    "academia.edu",
    "core.ac.uk",
    "medium.com",
    "researchgate.net",
    "scribd.com",
)
_NON_SOURCE_HOSTS = (
    "etsy.com",
    "pinterest.com",
)
_BENCHMARK_LEAK_MARKERS = (
    "/datasets/gaia/",
    "gaia-benchmark",
    "gaia_benchmark",
    "gaia benchmark",
    "harbor-datasets",
    "metadata.jsonl",
)
_BOT_PAGE_MARKERS = (
    "making sure you're not a bot",
    "verify you are human",
    "checking your browser",
    "captcha",
    "anubis",
    "cloudflare ray id",
)


def _truncate(text: str, limit: int = MAX_TOOL_TEXT) -> str:
    return text if len(text) <= limit else text[:limit] + "\n[truncated]"


def reset_research_session() -> None:
    """Clear per-question duplicate-search tracking."""

    session_id = uuid4().hex
    _SEARCH_SESSION.set(session_id)
    with _SEARCH_HISTORY_LOCK:
        _SEARCH_HISTORY[session_id] = set()
        while len(_SEARCH_HISTORY) > 256:
            _SEARCH_HISTORY.pop(next(iter(_SEARCH_HISTORY)))


def set_task_question(question: str) -> None:
    """Expose the current question to deterministic tool-input validators."""

    _TASK_QUESTION.set(question)


def _is_benchmark_leak_result(url: str, title: str = "", snippet: str = "") -> bool:
    """Reject pages that reproduce benchmark prompts or reference answers."""

    combined = f"{url} {title} {snippet}".lower()
    if any(marker in combined for marker in _BENCHMARK_LEAK_MARKERS):
        return True
    if bool(
        re.search(r"\bgaia\b", combined)
        and re.search(r"\b(?:task[_ -]?id|final answer|instruction\.md)\b", combined)
    ):
        return True

    # Search-poisoning pages often copy the complete task wording into an
    # unrelated page. Require a near-verbatim normalized copy so genuine source
    # abstracts that happen to share subject words are not excluded.
    question = _TASK_QUESTION.get().lower()
    if len(question) >= 80:
        normalized_question = " ".join(re.findall(r"[a-z0-9]+", question))
        normalized_snippet = " ".join(re.findall(r"[a-z0-9]+", snippet.lower()))
        if normalized_question and normalized_question in normalized_snippet:
            return True
    return False


def _is_unusable_search_result(item: dict[str, object]) -> bool:
    """Identify search pages, backend errors, and non-source result records."""

    url = str(item.get("href", "")).strip()
    title = str(item.get("title", "")).strip()
    snippet = str(item.get("body", "")).strip()
    if not url or _is_benchmark_leak_result(url, title, snippet):
        return True
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    path = parsed.path.lower()
    if any(
        host == domain or host.endswith(f".{domain}")
        for domain in _NON_SOURCE_HOSTS
    ):
        return True
    if (
        (host == "google.com" or host.endswith(".google.com"))
        and path.startswith("/search")
    ) or (
        host in {"bing.com", "www.bing.com", "search.yahoo.com", "search.brave.com"}
        and path.startswith("/search")
    ) or (
        host in {"duckduckgo.com", "www.duckduckgo.com"}
        and bool(parse_qs(parsed.query).get("q"))
    ):
        return True
    lowered_snippet = snippet.lower()
    return any(
        marker in lowered_snippet
        for marker in ("[error 500]", "internal server error", "bad gateway")
    )


def _source_priority(url: str) -> tuple[int, str]:
    """Estimate whether a result is a primary/publisher source."""

    host = (urlparse(url).hostname or "").lower()
    score = 0
    label = "general web"
    if _is_benchmark_leak_result(url):
        return -100, "excluded benchmark material"
    if any(host == domain or host.endswith(f".{domain}") for domain in _AGGREGATOR_HOSTS):
        return -10, "third-party mirror"
    if host.endswith((".gov", ".gov.uk", ".edu", ".ac.uk")):
        score += 5
        label = "likely official/academic"
    if any(term in host for term in ("journal", "journals", "publisher", "press")):
        score += 4
        label = "likely publisher"
    if host == "doi.org" or host.endswith(".doi.org"):
        score += 4
        label = "publication record"
    if _looks_like_pdf_url(url):
        score += 1
    return score, label


def _search_relevance(item: dict[str, object], query: str) -> int:
    """Score lexical relevance while penalizing generic navigation pages."""

    ignored = {
        "about", "after", "article", "from", "journal", "only", "that",
        "their", "this", "what", "when", "where", "which", "with",
    }
    query_terms = {
        token.rstrip("s")
        for token in re.findall(r"[a-z0-9]+", query.lower())
        if len(token) >= 4 and token not in ignored
    }
    title = str(item.get("title", "")).lower()
    snippet = str(item.get("body", "")).lower()
    title_terms = {token.rstrip("s") for token in re.findall(r"[a-z0-9]+", title)}
    snippet_terms = {
        token.rstrip("s") for token in re.findall(r"[a-z0-9]+", snippet)
    }
    score = 3 * len(query_terms & title_terms)
    score += min(5, len(query_terms & snippet_terms))
    if re.search(r"\b(editorial|archive|contents|database|home ?page)\b", title):
        score -= 7
    return score


def _source_candidate_score(item: dict[str, object], query: str) -> int:
    """Combine authority, relevance, and URL/title consistency."""

    url = str(item.get("href", ""))
    title = str(item.get("title", ""))
    source_score, _ = _source_priority(url)
    direct_pdf = _looks_like_pdf_url(url)
    score = 5 * source_score + _search_relevance(item, query)
    score += 3 * int(direct_pdf)
    score += 2 * int("/article/" in url.lower() or "/paper/" in url.lower())

    # Search-poisoning results sometimes copy the query into their title/snippet
    # while pointing to an unrelated, descriptively named PDF. Demote only
    # low-authority PDFs with enough path text to prove the mismatch; numeric or
    # opaque PDF paths and publisher/academic hosts are left unchanged.
    if direct_pdf and source_score <= 1:
        ignored_path_terms = {
            "article", "default", "document", "download", "files", "media",
            "paper", "pdfs", "source",
        }
        path_terms = {
            token.rstrip("s")
            for token in re.findall(r"[a-z]+", unquote(urlparse(url).path).lower())
            if len(token) >= 5 and token not in ignored_path_terms
        }
        title_terms = {
            token.rstrip("s")
            for token in re.findall(r"[a-z]+", title.lower())
            if len(token) >= 5
        }
        if len(path_terms) >= 3 and len(title_terms) >= 3 and not (
            path_terms & title_terms
        ):
            score -= 18
    return score


def _ranked_source_candidates(
    search_results: str, query: str
) -> list[tuple[int, int, str, str]]:
    """Parse formatted search output into relevance-ranked source candidates."""

    candidates: list[tuple[int, int, str, str]] = []
    for entry in re.split(r"\n\s*\n", search_results):
        url_match = re.search(r"^URL:\s*(https?://\S+)", entry, re.MULTILINE)
        priority_match = re.search(r"^Source priority:\s*(.+)$", entry, re.MULTILINE)
        if not url_match:
            continue
        url = url_match.group(1).rstrip(".,;)]}")
        if _is_benchmark_leak_result(url, entry):
            continue
        _, inferred_priority = _source_priority(url)
        priority = priority_match.group(1).strip() if priority_match else inferred_priority
        if priority in {"third-party mirror", "excluded benchmark material"}:
            continue
        title_match = re.search(r"^Title:\s*(.+)$", entry, re.MULTILINE)
        snippet_match = re.search(r"^Snippet:\s*(.*)$", entry, re.MULTILINE)
        item = {
            "title": title_match.group(1) if title_match else "",
            "body": snippet_match.group(1) if snippet_match else "",
            "href": url,
        }
        if _is_unusable_search_result(item):
            continue
        direct_pdf = int(_looks_like_pdf_url(url))
        candidates.append(
            (
                _source_candidate_score(item, query),
                direct_pdf,
                url,
                priority,
            )
        )
    return sorted(candidates, key=lambda item: (item[0], item[1]), reverse=True)


def _source_url_from_search_result(search_results: str, query: str = "") -> str:
    """Return the most relevant usable primary-source URL from formatted results."""

    urls = _source_urls_from_search_result(search_results, query)
    return urls[0] if urls else ""


def _source_urls_from_search_result(
    search_results: str, query: str = ""
) -> list[str]:
    """Return every usable source URL in ranked order without duplicates."""

    urls: list[str] = []
    for _, _, url, _ in _ranked_source_candidates(search_results, query):
        if url not in urls:
            urls.append(url)
    return urls


def _looks_like_pdf_url(url: str) -> bool:
    """Recognize common direct-download URLs, even without a .pdf suffix."""

    path = urlparse(url).path.lower()
    if any(
        path.endswith(suffix)
        for suffix in (".csv", ".htm", ".html", ".json", ".txt", ".xml")
    ):
        return False
    return path.endswith(".pdf") or bool(
        re.search(r"/(?:download|pdf)(?:/|$)", path)
    )


def _simplify_search_query(query: str) -> str:
    """Remove syntax that is inconsistently supported across DDGS backends."""

    simplified = re.sub(r"\s+\bOR\b\s+", " ", query, flags=re.IGNORECASE)
    simplified = re.sub(r"[()]", " ", simplified)
    return " ".join(simplified.split())


def _ddgs_text_search(query: str) -> list[dict[str, object]]:
    """Try DDGS auto-routing, then explicit independent search backends."""

    attempts = [
        (query, "auto"),
        (query, "duckduckgo,brave,google"),
    ]
    simplified = _simplify_search_query(query)
    if simplified.lower() != query.strip().lower():
        attempts.append((simplified, "duckduckgo,brave,google"))

    errors: list[str] = []
    for attempted_query, backend in attempts:
        try:
            results = list(
                DDGS(timeout=HTTP_TIMEOUT).text(
                    attempted_query,
                    max_results=8,
                    backend=backend,
                )
            )
        except Exception as exc:  # noqa: BLE001 - DDGS wraps backend failures.
            errors.append(f"{backend}: {type(exc).__name__}")
            continue
        usable_results = [
            item for item in results if not _is_unusable_search_result(item)
        ]
        if usable_results:
            return usable_results
        if results:
            errors.append(f"{backend}: only copied or unusable results")
    detail = ", ".join(errors) or "all backends returned no results"
    raise RuntimeError(f"DDGS search failed after fallback attempts ({detail}).")


def _existing_temp_file(file_path: str) -> Path:
    """Only allow tools to inspect temporary files supplied by the app."""

    path = Path(file_path).resolve()
    temp_root = Path(tempfile.gettempdir()).resolve()
    try:
        path.relative_to(temp_root)
    except ValueError as exc:
        raise ValueError("Attachment tools only accept files in the temp directory.") from exc
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _official_course_file(file_path: str) -> Path:
    """Restrict executable attachments to the runner's course-download folder."""

    path = _existing_temp_file(file_path)
    if not any(part.startswith("gaia_attachments_") for part in path.parts):
        raise ValueError(
            "For safety, Python execution is allowed only for attachments "
            "downloaded by the official evaluation runner."
        )
    return path


def _validate_public_url(url: str) -> None:
    """Reject local/private targets before the research tool makes a request."""

    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Only public http and https URLs are supported.")
    try:
        addresses = {
            item[4][0] for item in socket.getaddrinfo(parsed.hostname, parsed.port)
        }
    except socket.gaierror as exc:
        raise ValueError(f"Could not resolve URL host: {parsed.hostname}") from exc
    for address in addresses:
        candidate = ip_address(address)
        if not candidate.is_global:
            raise ValueError("Local, private, and reserved network URLs are blocked.")


@tool
def search_web(query: str) -> str:
    """Search once per query and rank likely primary sources ahead of mirrors.

    Args:
        query: A precise search query, including dates or quoted phrases when useful.
    """

    normalized_query = " ".join(query.lower().split())
    session_id = _SEARCH_SESSION.get()
    with _SEARCH_HISTORY_LOCK:
        previous_queries = _SEARCH_HISTORY.setdefault(session_id, set())
        if normalized_query in previous_queries:
            return (
                "DUPLICATE SEARCH BLOCKED: this exact query was already used for "
                "this question. Reformulate it with a rare phrase, title, author, "
                "publisher domain, or filetype:pdf."
            )
        previous_queries.add(normalized_query)

    try:
        raw_results = _ddgs_text_search(query)
    except Exception as exc:  # noqa: BLE001 - return a tool observation for any backend.
        return f"No search results found after DDGS fallbacks: {exc}"
    results = [
        item
        for item in raw_results
        if not _is_unusable_search_result(item)
    ]
    if not results:
        return (
            "No usable search results found. Results reproducing benchmark "
            "questions or answers were excluded; reformulate toward a primary source."
        )
    ranked_results = sorted(
        enumerate(results),
        key=lambda pair: (
            _source_candidate_score(pair[1], query),
            -pair[0],
        ),
        reverse=True,
    )
    blocks = []
    for _, item in ranked_results:
        url = str(item.get("href", ""))
        _, priority_label = _source_priority(url)
        blocks.append(
            f"Source priority: {priority_label}\n"
            f"Title: {item.get('title', '')}\n"
            f"URL: {url}\n"
            f"Snippet: {item.get('body', '')}"
        )
    return _truncate("\n\n".join(blocks))


@tool
def fetch_webpage(url: str) -> str:
    """Download a public HTML page and return readable text and PDF links.

    Args:
        url: An http or https URL found during research.
    """

    if _is_benchmark_leak_result(url):
        return (
            "BLOCKED BENCHMARK RESULT: this page appears to reproduce evaluation "
            "material. Find an independent primary source instead."
        )

    current_url = url
    response = None
    for _ in range(4):
        _validate_public_url(current_url)
        response = requests.get(
            current_url,
            timeout=HTTP_TIMEOUT,
            headers={"User-Agent": "Mozilla/5.0 GAIA-course-agent/1.0"},
            allow_redirects=False,
        )
        if response.is_redirect or response.is_permanent_redirect:
            location = response.headers.get("Location")
            response.close()
            if not location:
                break
            current_url = urljoin(current_url, location)
            continue
        break
    if response is None:
        raise RuntimeError("No web response was received.")
    if response.is_redirect or response.is_permanent_redirect:
        raise ValueError("Page redirected too many times.")
    response.raise_for_status()
    content_type = response.headers.get("Content-Type", "").lower()
    if "application/pdf" in content_type or response.content.startswith(b"%PDF"):
        return f"This URL is a PDF. Read it with fetch_pdf: {current_url}"
    if len(response.content) > 3_000_000:
        raise ValueError("Page is too large; find a smaller source.")
    soup = BeautifulSoup(response.text, "html.parser")
    pdf_links: list[str] = []
    for meta_name in ("citation_pdf_url", "eprints.document_url"):
        for meta in soup.find_all("meta", attrs={"name": meta_name}):
            content = str(meta.get("content", "")).strip()
            absolute_url = urljoin(current_url, content)
            if content and absolute_url not in pdf_links:
                pdf_links.append(absolute_url)
    for anchor in soup.find_all("a", href=True):
        href = str(anchor.get("href", ""))
        label = anchor.get_text(" ", strip=True)
        absolute_url = urljoin(current_url, href)
        parsed = urlparse(absolute_url)
        looks_like_pdf = parsed.path.lower().endswith(".pdf") or "pdf" in label.lower()
        if (
            looks_like_pdf
            and parsed.scheme in {"http", "https"}
            and absolute_url not in pdf_links
        ):
            pdf_links.append(absolute_url)
        if len(pdf_links) == 10:
            break
    for tag in soup(["script", "style", "nav", "footer", "noscript"]):
        tag.decompose()
    text = "\n".join(
        line.strip() for line in soup.get_text("\n").splitlines() if line.strip()
    )
    lowered_text = text.lower()
    if any(marker in lowered_text for marker in _BOT_PAGE_MARKERS):
        return (
            f"BLOCKED PAGE: {current_url} displayed a bot-verification page, "
            "not source evidence. Search for the exact title on an alternate "
            "official domain or for a direct PDF URL."
        )
    if pdf_links:
        text += "\n\nDirect PDF links found in page metadata or links:\n" + "\n".join(
            pdf_links[:10]
        )
    return _truncate(text)


@tool
def fetch_pdf(url: str) -> str:
    """Download a public PDF and extract page-labelled text for research.

    Args:
        url: A direct public http or https URL for a PDF document.
    """

    if _is_benchmark_leak_result(url):
        return (
            "BLOCKED BENCHMARK RESULT: this file appears to reproduce evaluation "
            "material. Find an independent primary source instead."
        )

    current_url = url
    response = None
    for _ in range(4):
        _validate_public_url(current_url)
        response = requests.get(
            current_url,
            timeout=HTTP_TIMEOUT,
            headers={"User-Agent": "Mozilla/5.0 GAIA-course-agent/1.0"},
            allow_redirects=False,
            stream=True,
        )
        if response.is_redirect or response.is_permanent_redirect:
            location = response.headers.get("Location")
            response.close()
            if not location:
                break
            current_url = urljoin(current_url, location)
            continue
        break
    if response is None:
        raise RuntimeError("No PDF response was received.")
    if response.is_redirect or response.is_permanent_redirect:
        response.close()
        raise ValueError("PDF redirected too many times.")
    try:
        if response.status_code in {401, 403, 429}:
            return (
                f"BLOCKED PDF (HTTP {response.status_code}): {current_url}. "
                "Do not guess from this failure. Open the publisher landing "
                "page to find its direct PDF, or search the exact title on an "
                "alternate official domain."
            )
        response.raise_for_status()
        declared_size = response.headers.get("Content-Length")
        if declared_size and int(declared_size) > MAX_PDF_BYTES:
            raise ValueError("PDF is too large; find a smaller or more specific source.")

        chunks: list[bytes] = []
        downloaded = 0
        for chunk in response.iter_content(chunk_size=64 * 1024):
            if not chunk:
                continue
            downloaded += len(chunk)
            if downloaded > MAX_PDF_BYTES:
                raise ValueError(
                    "PDF is too large; find a smaller or more specific source."
                )
            chunks.append(chunk)
        content = b"".join(chunks)
    finally:
        response.close()

    if not content.startswith(b"%PDF"):
        raise ValueError("The URL did not return a PDF document.")

    from pypdf import PdfReader

    reader = PdfReader(BytesIO(content), strict=False)
    page_blocks = []
    for page_number, page in enumerate(reader.pages[:MAX_PDF_PAGES], start=1):
        extracted = (page.extract_text() or "").strip()
        if extracted:
            page_blocks.append(f"[Page {page_number}]\n{extracted}")
    if not page_blocks:
        return (
            "The PDF has no extractable text. It may be scanned and require an "
            "OCR or vision tool."
        )
    if len(reader.pages) > MAX_PDF_PAGES:
        page_blocks.append(
            f"[Only the first {MAX_PDF_PAGES} of {len(reader.pages)} pages were read.]"
        )
    return _truncate("\n\n".join(page_blocks))


def _run_search_tool(query: str) -> str:
    return search_web.invoke({"query": query})


def _run_webpage_tool(url: str) -> str:
    return fetch_webpage.invoke({"url": url})


def _run_pdf_tool(url: str) -> str:
    return fetch_pdf.invoke({"url": url})


@tool
def research_primary_source(query: str) -> str:
    """Search, open, and read a likely primary source in one research action.

    This combines the normal search -> publisher page -> direct PDF sequence so
    small local models do not need a separate generation for every navigation
    step. If the first official-looking page is blocked, it tries alternatives.

    Args:
        query: A precise query using the exact title, author, or rare phrase.
    """

    search_results = _run_search_tool(query)
    if search_results.startswith(("DUPLICATE SEARCH BLOCKED", "No search results")):
        return search_results

    candidates = _ranked_source_candidates(search_results, query)

    attempted: list[str] = []
    readable_pages: list[tuple[int, str, str, str]] = []
    for score, _, url, priority in candidates:
        attempted.append(url)
        if len(attempted) > 5:
            break
        if _looks_like_pdf_url(url):
            try:
                pdf_text = _run_pdf_tool(url)
            except Exception as exc:  # noqa: BLE001 - try the next source candidate.
                attempted[-1] += f" ({type(exc).__name__})"
                continue
            if not _observation_is_blocked(pdf_text):
                return _truncate(f"Selected direct source PDF: {url}\n\n{pdf_text}")
            continue
        try:
            page_text = _run_webpage_tool(url)
        except Exception as exc:  # noqa: BLE001 - try the next source candidate.
            attempted[-1] += f" ({type(exc).__name__})"
            continue
        if page_text.startswith("BLOCKED PAGE"):
            continue
        if page_text.startswith("This URL is a PDF. Read it with fetch_pdf: "):
            pdf_url = page_text.removeprefix(
                "This URL is a PDF. Read it with fetch_pdf: "
            ).strip()
            try:
                pdf_text = _run_pdf_tool(pdf_url)
            except Exception:  # noqa: BLE001,S112 - try the next discovered PDF.
                continue
            if not _observation_is_blocked(pdf_text):
                return _truncate(
                    f"Selected source ({priority}): {pdf_url}\n\n{pdf_text}"
                )

        marker = "Direct PDF links found in page metadata or links:"
        if marker in page_text:
            pdf_section = page_text.split(marker, maxsplit=1)[1]
            pdf_urls = [
                candidate.rstrip(".,;)]}")
                for candidate in re.findall(r"https?://\S+", pdf_section)
                if not _is_benchmark_leak_result(candidate)
            ]
            source_host = (urlparse(url).hostname or "").lower()
            pdf_urls.sort(
                key=lambda candidate: (
                    (urlparse(candidate).hostname or "").lower() == source_host,
                    _looks_like_pdf_url(candidate),
                ),
                reverse=True,
            )
            for pdf_url in pdf_urls[:3]:
                try:
                    pdf_text = _run_pdf_tool(pdf_url)
                except Exception:  # noqa: BLE001,S112 - try the next discovered PDF.
                    continue
                if not _observation_is_blocked(pdf_text):
                    return _truncate(
                        f"Selected publisher page: {url}\n"
                        f"Selected direct PDF: {pdf_url}\n\n{pdf_text}"
                    )

        if len(page_text) >= 500:
            readable_pages.append((score, priority, url, page_text))

    if readable_pages:
        _, priority, url, page_text = max(readable_pages, key=lambda item: item[0])
        return _truncate(f"Selected source ({priority}): {url}\n\n{page_text}")

    attempted_text = "\n".join(attempted) or "none"
    return (
        "PRIMARY SOURCE NOT OPENED. Tried:\n"
        f"{attempted_text}\nReformulate with the exact title plus filetype:pdf "
        "or an alternate publisher domain."
    )


def _observation_is_blocked(text: str) -> bool:
    lowered = text.lower()
    return any(
        marker in lowered
        for marker in (
            "blocked page",
            "blocked pdf",
            "blocked benchmark result",
            "no extractable text",
            "error:",
        )
    )


@tool
def search_wikipedia(query: str) -> str:
    """Search English Wikipedia and return short extracts from up to three pages.

    Args:
        query: The subject, person, event, or phrase to find on Wikipedia.
    """

    response = requests.get(
        "https://en.wikipedia.org/w/api.php",
        params={
            "action": "query",
            "generator": "search",
            "gsrsearch": query,
            "gsrlimit": 3,
            "prop": "extracts|info",
            "exintro": 1,
            "explaintext": 1,
            "inprop": "url",
            "format": "json",
        },
        timeout=HTTP_TIMEOUT,
    )
    response.raise_for_status()
    pages = response.json().get("query", {}).get("pages", {}).values()
    blocks = [
        f"Title: {page.get('title', '')}\nURL: {page.get('fullurl', '')}\n{page.get('extract', '')}"
        for page in pages
    ]
    return _truncate("\n\n".join(blocks)) or "No Wikipedia pages found."


_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


class CountLineItem(BaseModel):
    """One explicit role/item and how many individuals it represents."""

    description: str = Field(
        description=(
            "Specific role or source phrase, such as 'brother and living spouse' "
            "or 'brother children who eat'"
        )
    )
    count: int = Field(
        ge=1,
        description="Number explicitly supported by this one role/source phrase",
    )


class AuditedQuantityGroup(BaseModel):
    """A rate category whose total is derived from explicit line items."""

    label: str = Field(description="Rate category, such as adults or eating children")
    included_items: list[CountLineItem] = Field(
        description=(
            "Mutually exclusive included roles. Do not include deceased, absent, "
            "or non-consuming people; record those in excluded_items instead."
        )
    )
    amount_per_member: float = Field(
        description="Base quantity consumed or contributed by each included person/item"
    )


def _evaluate_math(node: ast.AST) -> float | int:
    if isinstance(node, ast.Expression):
        return _evaluate_math(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        return _BINARY_OPERATORS[type(node.op)](
            _evaluate_math(node.left), _evaluate_math(node.right)
        )
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        return _UNARY_OPERATORS[type(node.op)](_evaluate_math(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        functions = {"sqrt": math.sqrt, "ceil": math.ceil, "floor": math.floor}
        if node.func.id in functions and len(node.args) == 1:
            return functions[node.func.id](_evaluate_math(node.args[0]))
    raise ValueError("Expression contains unsupported syntax.")


@tool
def calculate(expression: str) -> str:
    """Evaluate arithmetic without running arbitrary Python code.

    Args:
        expression: Arithmetic using numbers, +, -, *, /, //, %, **, parentheses,
            and optionally sqrt(), ceil(), or floor().
    """

    tree = ast.parse(expression, mode="eval")
    return str(_evaluate_math(tree))


_QUANTITY_TOKEN_VALUES = {
    "half": 0.5,
    "one half": 0.5,
    "quarter": 0.25,
    "one quarter": 0.25,
    "one-third": 1 / 3,
    "one third": 1 / 3,
}
_MEASUREMENT_UNIT = (
    r"pounds?|lbs?|kilograms?|kgs?|ounces?|oz|grams?|g|liters?|litres?|"
    r"milliliters?|millilitres?|ml|meters?|metres?|m|centimeters?|centimetres?|cm"
)


def _quantity_token_value(token: str) -> float:
    normalized = " ".join(token.lower().split())
    if normalized in _QUANTITY_TOKEN_VALUES:
        return _QUANTITY_TOKEN_VALUES[normalized]
    if "/" in normalized:
        numerator, denominator = normalized.split("/", maxsplit=1)
        return float(numerator) / float(denominator)
    return float(normalized)


def _question_quantity_input_issues(
    converted_units_per_base_unit: float,
    package_size_in_converted_units: float,
) -> list[str]:
    """Validate explicit conversion and package facts from the active question."""

    question = _TASK_QUESTION.get().lower()
    if not question:
        return []
    issues: list[str] = []
    amount_pattern = (
        r"half|one\s+half|quarter|one\s+quarter|one[- ]third|"
        r"\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?"
    )
    conversion_patterns = (
        (
            rf"\b(?:the\s+)?average\s+[a-z][a-z -]{{0,35}}?\s+"
            rf"(?:is|weighs?|equals?|contains?)\s+(?:about\s+|approximately\s+)?"
            rf"(?P<amount>{amount_pattern})\s+(?:an?\s+)?(?P<unit>{_MEASUREMENT_UNIT})\b"
        ),
        (
            rf"\b(?:each|one)\s+[a-z][a-z -]{{0,30}}?\s+"
            rf"(?:is|weighs?|equals?|contains?)\s+(?:about\s+|approximately\s+)?"
            rf"(?P<amount>{amount_pattern})\s+(?:an?\s+)?(?P<unit>{_MEASUREMENT_UNIT})\b"
        ),
    )
    stated_conversions: list[float] = []
    for pattern in conversion_patterns:
        for match in re.finditer(pattern, question, re.IGNORECASE):
            stated_conversions.append(_quantity_token_value(match.group("amount")))
    if stated_conversions:
        stated_conversion = stated_conversions[0]
        if all(
            math.isclose(value, stated_conversion, rel_tol=1e-9)
            for value in stated_conversions
        ) and not math.isclose(
            converted_units_per_base_unit,
            stated_conversion,
            rel_tol=1e-9,
        ):
            issues.append(
                "The question explicitly states a per-item conversion of "
                f"{stated_conversion:g}; converted_units_per_base_unit cannot be "
                f"{converted_units_per_base_unit:g}."
            )

    package_match = re.search(
        r"\bsold in\s+(\d+(?:\.\d+)?)\s*[- ]?"
        rf"(?:{_MEASUREMENT_UNIT})[ -]"
        r"(?:bags?|packs?|packages?|boxes?|containers?)\b",
        question,
    )
    if package_match:
        stated_size = float(package_match.group(1))
        if not math.isclose(
            package_size_in_converted_units,
            stated_size,
            rel_tol=1e-9,
        ):
            issues.append(
                f"The question explicitly states a package size of {stated_size:g}; "
                "package_size_in_converted_units cannot be "
                f"{package_size_in_converted_units:g}."
            )
    return issues


@tool
def calculate_audited_quantity(
    groups: list[AuditedQuantityGroup],
    excluded_items: list[CountLineItem],
    converted_units_per_base_unit: float = 1.0,
    package_size_in_converted_units: float = 1.0,
    round_up: bool = False,
) -> str:
    """Calculate category totals from explicit, mutually exclusive count lines.

    Use for grouped word problems. Break each category into source-supported
    roles instead of hiding the reasoning in a guessed total. Excluded line
    items are audit notes and must not also appear in included groups.

    Args:
        groups: Rate groups containing explicit included role/count line items.
        excluded_items: Explicit nonparticipants and their counts, for auditing.
        converted_units_per_base_unit: Converted amount for one base unit.
        package_size_in_converted_units: Amount contained by one package.
        round_up: Whether a fractional package count is rounded upward.
    """

    if not groups:
        raise ValueError("At least one audited group is required.")
    if converted_units_per_base_unit <= 0 or package_size_in_converted_units <= 0:
        raise ValueError("Unit conversion and package size must be positive.")

    parsed_groups = [
        group
        if isinstance(group, AuditedQuantityGroup)
        else AuditedQuantityGroup.model_validate(group)
        for group in groups
    ]
    parsed_exclusions = [
        item if isinstance(item, CountLineItem) else CountLineItem.model_validate(item)
        for item in excluded_items
    ]
    input_issues = _question_quantity_input_issues(
        converted_units_per_base_unit,
        package_size_in_converted_units,
    )
    if input_issues:
        raise ValueError(" ".join(input_issues))
    excluded_descriptions = {
        " ".join(item.description.lower().split()) for item in parsed_exclusions
    }
    seen_descriptions: set[str] = set()
    ledger_lines: list[str] = []
    base_total = 0.0
    for group in parsed_groups:
        if not group.included_items:
            raise ValueError(f"Group '{group.label}' has no included count lines.")
        group_count = 0
        for item in group.included_items:
            normalized = " ".join(item.description.lower().split())
            if normalized in seen_descriptions:
                raise ValueError(
                    f"Count line is duplicated across included groups: {item.description}"
                )
            if normalized in excluded_descriptions:
                raise ValueError(
                    f"Count line appears as both included and excluded: {item.description}"
                )
            seen_descriptions.add(normalized)
            group_count += item.count
        subtotal = group_count * group.amount_per_member
        base_total += subtotal
        basis = "; ".join(
            f"{item.description}={item.count}" for item in group.included_items
        )
        ledger_lines.append(
            f"{group.label}: ({basis}) -> {group_count} x "
            f"{group.amount_per_member:g} = {subtotal:g}"
        )

    converted_total = base_total * converted_units_per_base_unit
    package_count = converted_total / package_size_in_converted_units
    final_result: float | int = math.ceil(package_count) if round_up else package_count
    exclusions = (
        "; ".join(f"{item.description}={item.count}" for item in parsed_exclusions)
        if parsed_exclusions
        else "none"
    )
    return (
        "Audited count ledger:\n"
        + "\n".join(ledger_lines)
        + f"\nExcluded before counting: {exclusions}"
        + f"\nBase total: {base_total:g}"
        + f"\nConverted total: {base_total:g} x "
        + f"{converted_units_per_base_unit:g} = {converted_total:g}"
        + f"\nPackages before rounding: {converted_total:g} / "
        + f"{package_size_in_converted_units:g} = {package_count:g}"
        + f"\nFinal result: {final_result:g}"
    )


def _summarize_dataframe(frame: pd.DataFrame, name: str) -> str:
    numeric = frame.select_dtypes(include="number")
    parts = [
        f"Table: {name}",
        f"Shape: {frame.shape[0]} rows x {frame.shape[1]} columns",
        f"Columns: {list(frame.columns)}",
        "First rows:\n" + frame.head(20).to_string(index=False),
    ]
    if not numeric.empty:
        parts.append("Numeric column sums:\n" + numeric.sum().to_string())
    return "\n".join(parts)


@tool
def inspect_attachment(file_path: str) -> str:
    """Read a benchmark attachment such as text, CSV, Excel, PDF, JSON, or DOCX.

    Args:
        file_path: The exact temporary path supplied in the task prompt.
    """

    path = _existing_temp_file(file_path)
    suffix = path.suffix.lower()

    if suffix in {".txt", ".md", ".py", ".json", ".jsonld"}:
        return _truncate(path.read_text(encoding="utf-8", errors="replace"))
    if suffix == ".csv":
        return _truncate(_summarize_dataframe(pd.read_csv(path), path.name))
    if suffix in {".xlsx", ".xls"}:
        sheets = pd.read_excel(path, sheet_name=None)
        return _truncate(
            "\n\n".join(
                _summarize_dataframe(frame, name) for name, frame in sheets.items()
            )
        )
    if suffix == ".pdf":
        from pypdf import PdfReader

        text = "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
        return _truncate(text)
    if suffix == ".docx":
        from docx import Document

        document = Document(path)
        return _truncate("\n".join(p.text for p in document.paragraphs))
    if suffix == ".zip":
        import zipfile

        with zipfile.ZipFile(path) as archive:
            return "Archive members:\n" + "\n".join(archive.namelist())
    if suffix in {".png", ".jpg", ".jpeg", ".webp"}:
        from PIL import Image

        with Image.open(path) as image:
            return (
                f"Image file {path.name}: size={image.size}, mode={image.mode}. "
                "This starter only reports metadata. Add a vision-model tool in "
                "tools.py to understand image contents."
            )
    if suffix in {".mp3", ".wav", ".flac", ".m4a"}:
        return "Audio attachment detected. Use transcribe_audio with this path."
    return f"Unsupported attachment type: {suffix or '[no extension]'}"


@tool
def run_python_attachment(file_path: str) -> str:
    """Run a trusted .py attachment from the official course API in isolated mode.

    Args:
        file_path: The exact temporary path to the official Python attachment.
    """

    path = _official_course_file(file_path)
    if path.suffix.lower() != ".py":
        raise ValueError("This tool only accepts .py attachments.")
    blocked_environment_keys = {
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "HF_TOKEN",
        "HUGGINGFACEHUB_API_TOKEN",
        "OPENAI_API_KEY",
    }
    child_environment = {
        key: value
        for key, value in os.environ.items()
        if key.upper() not in blocked_environment_keys
    }
    result = subprocess.run(
        [sys.executable, "-I", str(path)],
        cwd=path.parent,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
        env=child_environment,
    )
    return _truncate(
        f"Exit code: {result.returncode}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )


@tool
def transcribe_audio(file_path: str) -> str:
    """Transcribe an audio attachment using a Hugging Face ASR model.

    Args:
        file_path: The exact temporary path to an audio attachment.
    """

    path = _existing_temp_file(file_path)
    token = os.getenv("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN is required for audio transcription.")
    model = os.getenv("GAIA_ASR_MODEL", "openai/whisper-large-v3")
    output = InferenceClient(token=token).automatic_speech_recognition(
        path,
        model=model,
    )
    return _truncate(output.text)


def _youtube_video_id(url_or_id: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", url_or_id):
        return url_or_id
    parsed = urlparse(url_or_id)
    if parsed.hostname in {"youtu.be", "www.youtu.be"}:
        return parsed.path.strip("/")
    if parsed.hostname and "youtube.com" in parsed.hostname:
        return parse_qs(parsed.query).get("v", [""])[0]
    return ""


@tool
def get_youtube_transcript(url_or_id: str) -> str:
    """Fetch the spoken transcript of a YouTube video when captions exist.

    Args:
        url_or_id: A YouTube URL or its 11-character video ID.
    """

    video_id = _youtube_video_id(url_or_id)
    if not video_id:
        raise ValueError("Could not identify the YouTube video ID.")
    from youtube_transcript_api import YouTubeTranscriptApi

    transcript = YouTubeTranscriptApi().fetch(video_id, languages=["en"])
    text = " ".join(snippet.text for snippet in transcript)
    return _truncate(text)


@tool
def analyze_youtube_video(url_or_id: str, question: str) -> str:
    """Use Gemini's native video understanding for a visual YouTube question.

    Use this instead of captions when the answer depends on what is visible,
    shown, featured, counted on camera, or happening simultaneously.

    Args:
        url_or_id: A public YouTube URL or its 11-character video ID.
        question: The exact visual question to answer from the video.
    """

    video_id = _youtube_video_id(url_or_id)
    if not video_id:
        raise ValueError("Could not identify the YouTube video ID.")
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key or "replace" in api_key.lower() or "your" in api_key.lower():
        raise ValueError("GEMINI_API_KEY is required for YouTube visual analysis.")

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError as exc:
        raise RuntimeError(
            "YouTube visual analysis needs langchain-google-genai."
        ) from exc

    model_name = os.getenv("GEMINI_VIDEO_MODEL", "").strip() or os.getenv(
        "GEMINI_MODEL", "gemini-3.8-flash"
    ).strip()
    model_kwargs: dict[str, object] = {
        "model": model_name,
        "api_key": api_key,
        "max_tokens": 700,
        "timeout": 180,
        "max_retries": 2,
    }
    if "flash-lite" not in model_name.lower():
        model_kwargs["temperature"] = (
            1.0 if "gemini-3" in model_name.lower() else 0.0
        )
    model = ChatGoogleGenerativeAI(**model_kwargs)
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    response = model.invoke(
        [
            HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": (
                            "Inspect the full video's visual content, not merely its "
                            "title, description, audio, or captions. For counts, scan "
                            "the timeline and count simultaneous visible instances at "
                            "each relevant moment. Answer the exact question below "
                            "concisely and include a short timestamp/evidence note so "
                            "the calling agent can verify it.\n\nQuestion: "
                            f"{question}"
                        ),
                    },
                    {
                        "type": "media",
                        "file_uri": video_url,
                        "mime_type": "video/mp4",
                    },
                ]
            )
        ]
    )
    content = response.content
    if isinstance(content, str):
        return _truncate(content)
    parts = [
        str(block.get("text", ""))
        for block in content
        if isinstance(block, dict) and block.get("text")
    ]
    return _truncate("\n".join(parts)) or "Gemini returned no video analysis text."


ALL_TOOLS = [
    search_web,
    research_primary_source,
    fetch_webpage,
    fetch_pdf,
    search_wikipedia,
    calculate,
    calculate_audited_quantity,
    inspect_attachment,
    run_python_attachment,
    transcribe_audio,
    get_youtube_transcript,
    analyze_youtube_video,
]

RESEARCH_TOOLS = [
    search_web,
    research_primary_source,
    fetch_webpage,
    fetch_pdf,
    search_wikipedia,
    calculate,
    calculate_audited_quantity,
    get_youtube_transcript,
    analyze_youtube_video,
]

CALCULATION_TOOLS = [calculate, calculate_audited_quantity]
