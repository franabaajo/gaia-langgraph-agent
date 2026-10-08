"""LangGraph agent used by both the local runner and the Gradio app.

This is the main file to edit while developing the assignment. The workflow
plans each task, routes it to a specialist, uses tools, and verifies the answer
before returning it.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from answer_normalize import normalize_answer, normalize_answer_for_question
from tools import (
    ALL_TOOLS,
    CALCULATION_TOOLS,
    RESEARCH_TOOLS,
    _looks_like_pdf_url,
    _source_urls_from_search_result,
    reset_research_session,
    set_task_question,
)

# Every backend must use a chat model with tool-calling support. Gemini is the
# hosted default; Hugging Face is an alternative and Ollama is for local use.
DEFAULT_BACKEND = "gemini"
DEFAULT_MODEL = "openai/gpt-oss-20b"
DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"
DEFAULT_GEMINI_FALLBACK_MODEL = "gemini-3.5-flash"
DEFAULT_GEMINI_THINKING_LEVEL = "low"
DEFAULT_OLLAMA_MODEL = "qwen3:8b"
DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
HF_ROUTER_URL = "https://router.huggingface.co/v1"
MAX_GRAPH_STEPS = 40
LEAN_MAX_GRAPH_STEPS = 16
LEAN_MAX_TOOL_ROUNDS = 2
LEAN_RESEARCH_MAX_TOOL_ROUNDS = 4
MAX_TOOL_ROUNDS = 3
MAX_VERIFICATION_REVISIONS = 1
MAX_VERIFIER_EVIDENCE = 7_000
TRACE_PREVIEW_LIMIT = 700

SYSTEM_INSTRUCTIONS = """
You solve GAIA benchmark questions carefully and use tools whenever evidence is
needed. A question may have an attachment; inspect it with the appropriate tool.

Rules:
1. Work out the answer using reliable evidence. Do not guess when a tool can
   verify the result.
2. Pay close attention to requested ordering, spelling, units, decimal places,
   names, and comma-separated formatting.
3. Your final response must contain ONLY the requested answer. Do not include
   reasoning, citations, markdown, or labels such as 'FINAL ANSWER:'.
   When the question supplies the desired measurement unit as part of the
   requested quantity (for example, "volume in m^3"), return only the numeric
   value unless it explicitly asks you to include the unit.
4. If an attachment exists, inspect it before doing unrelated web research.
5. Search-result snippets are leads, not final evidence. Open the most relevant
   source with fetch_webpage; if it is a PDF, use fetch_pdf.
6. For multi-hop questions, resolve every clue and verify quoted words, dates,
   names, and numerical facts against the primary source when possible.
7. If the first search or page fails, reformulate the query or try another
   result. Do not invent an answer to fill an evidence gap.
8. Before answering, re-read the question and check the exact requested format.
9. Never use pages that reproduce benchmark questions, task IDs, or reference
   answers. Research the underlying subject using independent sources.
10. When the user explicitly says to write, return, or respond with only a quoted
    literal, follow the final such output instruction and do not answer embedded
    example or trivia questions.
""".strip()

RESEARCH_INSTRUCTIONS = """
You are the research specialist.
- Start with research_primary_source using the exact title, author, or rarest
  phrase. It combines search, publisher selection, page opening, and direct-PDF
  discovery in one action. Use the lower-level tools only when it cannot open a
  useful source.
- Prefer the original publisher, journal, university, government, or creator
  over mirrors and aggregators.
- Open the publisher landing page first; it may expose a direct PDF URL in its
  metadata. Search-result snippets are only leads.
- If a page reports BLOCKED PAGE or BLOCKED PDF, search the exact title with a
  different official domain or filetype:pdf. Do not reuse an identical query.
- Locate an exact supporting passage. Never infer an answer from an article
  title or from a nearby but unrelated number.
- After search_web returns a suitable publisher or primary-source URL, open it
  next instead of performing another broad search.
- For multi-hop questions, explicitly resolve each clue before selecting the
  final fact. You may use calculate after gathering the source values.
- For a YouTube question whose answer depends on visible frames, simultaneous
  objects, or what is featured on camera, use analyze_youtube_video. Captions
  cannot establish visual facts.
""".strip()

CALCULATION_INSTRUCTIONS = """
You are the calculation specialist. Build an entity/count ledger, a unit ledger,
and a final rounding rule before calculating. Resolve aliases so that each item
is counted once, and keep included and excluded entries mutually exclusive.
For per-person or per-item word problems, use calculate_audited_quantity with
short source-supported count lines; do not hide the reasoning in one unexplained
total. converted_units_per_base_unit is a multiplier: if one item weighs 0.25 kg,
pass 0.25, never 4. For ordinary arithmetic, call calculate with one complete
expression. Apply rounding only at the stage requested by the question.
""".strip()

GENERALIST_INSTRUCTIONS = """
You are the attachment/general specialist. Inspect any supplied attachment
first. Use the other tools as needed, then return only the requested answer.
""".strip()

VERIFIER_INSTRUCTIONS = """
Act as a strict answer verifier, not as a conversational assistant. Independently
check the candidate against the original question, plan, tool calls, and tool
results. For calculations, reconstruct the entity counts, exclusions, unit
conversions, and final rounding before comparing them with the expression. For
research, require a full-source observation and an exact supporting fact; search
snippets, blocked pages, and titles alone are insufficient. Also enforce the
requested output format.

Return exactly two lines:
VERDICT: APPROVE or VERDICT: REVISE
FEEDBACK: a short actionable reason. If revising a calculation, include the
corrected counts, exclusions, unit chain, and corrected expression or grouped
ledger; never merely say that a count is wrong.
""".strip()

LEAN_INSTRUCTIONS = """
You are running in the low-latency profile. Solve the task in as few model
turns as possible. Call at most one tool at a time. For research, begin with
research_primary_source instead of alternating between search and page-fetch
tools. For population or grouped arithmetic, use calculate_audited_quantity
once and break every category into short, mutually exclusive source-supported
count lines. Put nonparticipants in excluded_items and do not infer unstated
members. For simple arithmetic, use calculate. Once a useful tool result is
available, answer directly instead of continuing exploratory tool calls.
Calculations and general questions have two tool rounds; research and mixed
tasks have four so a failed search can be reformulated before a source is opened.
""".strip()


class AgentState(MessagesState, total=False):
    """State shared by the visible planning, specialist, and verifier nodes."""

    question: str
    task_id: str
    attachment_path: str
    task_type: str
    plan: list[str]
    active_worker: str
    verification_status: str
    verification_revisions: int
    tool_rounds: int
    final_answer: str


class AgentConfigurationError(RuntimeError):
    """Raised when a required secret or setting is missing or invalid."""


def _token_is_configured(token: str | None) -> bool:
    return bool(token and token.startswith("hf_") and "your_read_token" not in token)


def _routed_model_id(model_id: str, provider: str | None) -> str:
    """Add Hugging Face's optional ``:provider`` suffix once."""

    provider = (provider or "").strip()
    if not provider or provider.lower() == "auto":
        return model_id
    if ":" in model_id.rsplit("/", maxsplit=1)[-1]:
        return model_id
    return f"{model_id}:{provider}"


def _selected_backend() -> str:
    """Return a normalized backend name and reject configuration typos."""

    backend = os.getenv("MODEL_BACKEND", DEFAULT_BACKEND).strip().lower()
    aliases = {"hf": "huggingface", "google": "gemini", "local": "ollama"}
    backend = aliases.get(backend, backend)
    if backend not in {"gemini", "huggingface", "ollama"}:
        raise AgentConfigurationError(
            "MODEL_BACKEND must be 'gemini', 'huggingface', or 'ollama'."
        )
    return backend


def _selected_agent_profile(backend: str | None = None) -> str:
    """Select a fast local graph or the fuller planning/verifier graph."""

    selected_backend = backend or _selected_backend()
    configured = os.getenv("AGENT_PROFILE", "").strip().lower()
    if not configured:
        return "lean" if selected_backend in {"gemini", "ollama"} else "robust"
    aliases = {"fast": "lean", "full": "robust"}
    profile = aliases.get(configured, configured)
    if profile not in {"lean", "robust"}:
        raise AgentConfigurationError(
            "AGENT_PROFILE must be either 'lean' or 'robust'."
        )
    return profile


def _env_bool(name: str, default: bool) -> bool:
    """Read a true/false environment setting with a useful error message."""

    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return default
    value = raw_value.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise AgentConfigurationError(
        f"{name} must be true or false, not {raw_value!r}."
    )


def _env_positive_int(name: str, default: int) -> int:
    """Read a positive integer environment setting."""

    raw_value = os.getenv(name, str(default))
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise AgentConfigurationError(f"{name} must be a whole number.") from exc
    if value <= 0:
        raise AgentConfigurationError(f"{name} must be greater than zero.")
    return value


def _gemini_thinking_level() -> str:
    """Read Gemini 3's bounded reasoning setting."""

    level = os.getenv(
        "GEMINI_THINKING_LEVEL", DEFAULT_GEMINI_THINKING_LEVEL
    ).strip().lower()
    if level not in {"low", "medium", "high"}:
        raise AgentConfigurationError(
            "GEMINI_THINKING_LEVEL must be low, medium, or high."
        )
    return level


def _gemini_sampling_kwargs(model_id: str) -> dict[str, float]:
    """Avoid passing sampling settings to models with fixed defaults."""

    return {} if "flash-lite" in model_id.lower() else {"temperature": 1.0}


def _gemini_key_is_configured(key: str | None) -> bool:
    """Reject blank/example keys without assuming one fixed key prefix."""

    if not key or not key.strip():
        return False
    lowered = key.lower()
    return "your" not in lowered and "replace" not in lowered


def _is_transient_gemini_error(exc: Exception) -> bool:
    """Return whether Google reported a temporary server/capacity failure."""

    detail = str(exc).lower()
    return any(
        marker in detail
        for marker in (
            "503",
            "504",
            "service_unavailable",
            "servererror",
            "high demand",
            "temporarily overloaded",
            "deadline_exceeded",
        )
    )


def _is_rate_limited_gemini_error(exc: Exception) -> bool:
    """Return whether Gemini rejected a request for quota/rate-limit reasons."""

    detail = str(exc).lower()
    return any(
        marker in detail
        for marker in ("429", "resource_exhausted", "quota", "rate limit", "rate-limit")
    )


def _is_retryable_gemini_error(exc: Exception) -> bool:
    return _is_transient_gemini_error(exc) or _is_rate_limited_gemini_error(exc)


def configuration_report() -> list[str]:
    """Explain which settings are ready without revealing secret values."""

    backend = _selected_backend()
    profile = _selected_agent_profile(backend)
    token_ready = _token_is_configured(os.getenv("HF_TOKEN"))
    report = [
        "OK - Agent orchestration uses LangGraph",
        f"OK - MODEL_BACKEND={backend}",
        f"OK - AGENT_PROFILE={profile}",
        "OK - Web search uses DDGS and needs no API key",
        "OK - Gemini video analysis is available for visual YouTube questions",
        "TIP - Put secrets in Space Settings > Repository secrets, never in code",
    ]
    if backend == "huggingface":
        provider = os.getenv("HF_INFERENCE_PROVIDER") or "auto (fastest available)"
        model = os.getenv("GAIA_MODEL", DEFAULT_MODEL)
        report[3:3] = [
            f"{'OK' if token_ready else 'MISSING'} - HF_TOKEN (required by the Hugging Face backend)",
            f"INFO - GAIA_MODEL={model}",
            f"INFO - HF_INFERENCE_PROVIDER={provider}",
        ]
    elif backend == "ollama":
        model = os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
        base_url = os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL)
        reasoning = _env_bool("OLLAMA_REASONING", False)
        context_length = _env_positive_int("OLLAMA_CONTEXT_LENGTH", 8192)
        num_predict = _env_positive_int("OLLAMA_NUM_PREDICT", 600)
        report[3:3] = [
            f"INFO - OLLAMA_MODEL={model}",
            f"INFO - OLLAMA_BASE_URL={base_url}",
            f"INFO - OLLAMA_REASONING={str(reasoning).lower()}",
            f"INFO - OLLAMA_CONTEXT_LENGTH={context_length}",
            f"INFO - OLLAMA_NUM_PREDICT={num_predict}",
            f"{'OK' if token_ready else 'OPTIONAL'} - HF_TOKEN (only needed for the audio transcription tool)",
            "TIP - Ollama must be running and the selected model must be pulled",
        ]
    else:
        key_ready = _gemini_key_is_configured(os.getenv("GEMINI_API_KEY"))
        model = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
        fallback_model = os.getenv(
            "GEMINI_FALLBACK_MODEL", DEFAULT_GEMINI_FALLBACK_MODEL
        ).strip()
        thinking_level = _gemini_thinking_level()
        max_tokens = _env_positive_int("GEMINI_MAX_OUTPUT_TOKENS", 1200)
        report[3:3] = [
            f"{'OK' if key_ready else 'MISSING'} - GEMINI_API_KEY (required by the Gemini backend)",
            f"INFO - GEMINI_MODEL={model}",
            f"INFO - GEMINI_FALLBACK_MODEL={fallback_model or 'disabled'}",
            f"INFO - GEMINI_THINKING_LEVEL={thinking_level}",
            f"INFO - GEMINI_MAX_OUTPUT_TOKENS={max_tokens}",
            f"{'OK' if token_ready else 'OPTIONAL'} - HF_TOKEN (only needed for the audio transcription tool)",
        ]
    return report


_RESEARCH_MARKERS = (
    "according to",
    "article",
    "author",
    "book",
    "journal",
    "paper",
    "published",
    "quoted",
    "source",
    "video",
    "website",
    "what word",
    "when did",
    "where did",
    "which ",
    "who ",
    "http://",
    "https://",
)
_CALCULATION_MARKERS = (
    "average",
    "calculate",
    "how many",
    "how much",
    "percent",
    "ratio",
    "total",
)
_FACTUAL_LOOKUP_PATTERNS = (
    r"^\s*(?:who|where|when|which)\b",
    r"^\s*what\s+(?:city|country|date|day|name|species|state|team|word|year)\b",
    (
        r"^\s*what\s+(?:is|was|are|were)\s+(?:the\s+)?"
        r"(?:capital|name|title|location|population|nationality|occupation|official|record)\b"
    ),
    r"\bhow many\b.{0,100}\b(?:did|has|had|was|were)\b",
)
_FULL_SOURCE_TOOLS = {
    "analyze_youtube_video",
    "fetch_pdf",
    "fetch_webpage",
    "get_youtube_transcript",
    "inspect_attachment",
    "research_primary_source",
    "search_wikipedia",
    "transcribe_audio",
}


def _explicit_literal_answer(question: str) -> str:
    """Extract the last unambiguous quoted `... only ...` output instruction."""

    patterns = (
        (
            r"\b(?:write|return|output|give)\s+only\s+(?:the\s+)?"
            r"(?:(?:word|phrase|text|answer)\s+)?[\"\u201c']([^\"\u201d'\r\n]+)[\"\u201d']"
        ),
        (
            r"\b(?:answer|respond)\s+(?:with\s+)?only\s+(?:the\s+)?"
            r"(?:(?:word|phrase|text|answer)\s+)?[\"\u201c']([^\"\u201d'\r\n]+)[\"\u201d']"
        ),
    )
    matches = [
        (match.start(), match.group(1))
        for pattern in patterns
        for match in re.finditer(pattern, question, re.IGNORECASE)
    ]
    return max(matches, default=(-1, ""), key=lambda item: item[0])[1].strip()


def _classify_task(question: str, has_attachment: bool = False) -> str:
    """Choose a specialist using transparent, deterministic signals."""

    if has_attachment:
        return "attachment"
    if _explicit_literal_answer(question):
        return "general"
    lowered = question.lower()
    needs_research = any(marker in lowered for marker in _RESEARCH_MARKERS)
    needs_calculation = any(marker in lowered for marker in _CALCULATION_MARKERS)
    factual_lookup = any(
        re.search(pattern, lowered, re.IGNORECASE | re.DOTALL)
        for pattern in _FACTUAL_LOOKUP_PATTERNS
    )
    explicit_computation = bool(
        re.search(
            r"\b(?:calculate|compute|total|average|percent|ratio|difference|"
            r"how many more|how many fewer|convert)\b|\d\s*[+*/^-]\s*\d",
            lowered,
        )
    )
    if factual_lookup:
        needs_research = True
        if not explicit_computation:
            needs_calculation = False
    if needs_research and needs_calculation:
        return "mixed"
    if needs_calculation:
        return "calculation"
    if needs_research:
        return "research"
    return "general"


def _build_plan(task_type: str) -> list[str]:
    """Create a short plan that the selected worker must follow."""

    if task_type == "attachment":
        return [
            "Inspect the supplied attachment before any unrelated research.",
            "Extract only the facts needed by the question.",
            "Use research or calculation tools if the attachment requires them.",
            "Check the requested answer format.",
        ]
    if task_type == "calculation":
        return [
            "Create an explicit entity/count ledger and resolve aliases.",
            "Keep included and excluded entries mutually exclusive.",
            "Create a unit-conversion ledger.",
            "Use the audited or free-form calculator that fits the problem.",
            "Independently verify counts, units, and final rounding.",
        ]
    if task_type == "mixed":
        return [
            "Identify the exact primary source and resolve every clue.",
            "Open the publisher page or direct PDF and capture source values.",
            "Convert units and calculate only after the source values are verified.",
            "Check that the final value and formatting answer the exact question.",
        ]
    if task_type == "research":
        return [
            "Resolve the named work, person, date, and other clues.",
            "Search for and open the most authoritative available source.",
            "Find an exact passage supporting the candidate answer.",
            "Check spelling, ordering, and requested output format.",
        ]
    return [
        "Determine the exact requested answer.",
        "Use a tool if any factual claim needs verification.",
        "Check the final output format.",
    ]


def _latest_answer_candidate(messages: list[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage) and not message.tool_calls:
            candidate = normalize_answer(_message_text(message))
            serialized_tool_call = re.search(
                r"<\s*/?\s*tool_call\s*>", candidate, re.IGNORECASE
            ) or (
                candidate.lstrip().startswith("{")
                and candidate.rstrip().endswith("}")
                and bool(re.search(r'"name"\s*:', candidate))
                and bool(re.search(r'"arguments"\s*:', candidate))
            )
            if serialized_tool_call:
                continue
            if (
                "search_web result:" in candidate.lower()
                and "source priority:" in candidate.lower()
                and "url:" in candidate.lower()
            ):
                continue
            return candidate
    for message in reversed(messages):
        if not isinstance(message, ToolMessage):
            continue
        text = _message_text(message).strip()
        if _observation_failed(text):
            continue
        if message.name == "calculate_audited_quantity":
            match = re.search(r"^Final result:\s*(.+)$", text, re.MULTILINE)
            if match:
                return normalize_answer(match.group(1))
        if message.name == "calculate" and re.fullmatch(
            r"[-+]?\d+(?:\.\d+)?", text
        ):
            return text
    return ""


def _observation_failed(text: str) -> bool:
    lowered = text.strip().lower()
    return any(
        marker in lowered
        for marker in (
            "blocked page:",
            "blocked pdf",
            "blocked benchmark result",
            "duplicate search blocked",
            "error:",
            "no extractable text",
            "no search results found",
            "no usable search results found",
            "primary source not opened",
        )
    )


def _successful_observations(
    messages: list[BaseMessage], tool_names: set[str] | None = None
) -> list[tuple[str, str]]:
    observations = []
    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        name = str(message.name or "tool")
        text = _message_text(message)
        if tool_names is not None and name not in tool_names:
            continue
        if not _observation_failed(text):
            observations.append((name, text))
    return observations


def _quoted_answer_from_evidence(
    question: str, messages: list[BaseMessage]
) -> str:
    """Extract a clearly supported quoted word from an opened full source."""

    lowered_question = question.lower()
    if not re.search(
        r"\bwhat\s+(?:single\s+)?word\b.{0,240}\bquot(?:e|ed)\b",
        lowered_question,
        re.DOTALL,
    ):
        return ""

    ignored = {
        "about", "after", "article", "author", "authors", "different",
        "from", "journal", "named", "quoted", "that", "their", "this",
        "what", "which", "with", "word",
    }
    question_terms = {
        token.rstrip("s")
        for token in re.findall(r"[a-z0-9]+", lowered_question)
        if len(token) >= 4 and token not in ignored
    }
    expects_two_sources = bool(
        re.search(r"\b(?:two|2)\b.{0,40}\b(?:authors?|writers?|scholars?|sources?)\b", lowered_question)
    )
    scores: dict[str, int] = {}
    display_values: dict[str, str] = {}
    for _, source in _successful_observations(messages, _FULL_SOURCE_TOOLS):
        for match in re.finditer(r"[\"\u201c]([^\"\u201d\r\n]{1,80})[\"\u201d]", source):
            value = match.group(1).strip(" \t.,;:!?()[]{}")
            words = re.findall(r"[A-Za-z][A-Za-z'-]*", value)
            if len(words) != 1 or len(value) > 40:
                continue
            normalized = value.casefold()
            if normalized in {"a", "an", "and", "or", "the"}:
                continue
            context = source[max(0, match.start() - 320) : match.end() + 320].lower()
            context_terms = {
                token.rstrip("s") for token in re.findall(r"[a-z0-9]+", context)
            }
            score = 2 * len(question_terms & context_terms)
            if expects_two_sources and re.search(r"\bboth\b", context):
                score += 4
            scores[normalized] = max(scores.get(normalized, 0), score)
            display_values.setdefault(normalized, value)

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    if not ranked or ranked[0][1] < 6:
        return ""
    if len(ranked) > 1 and ranked[0][1] - ranked[1][1] < 3:
        return ""
    return display_values[ranked[0][0]]


def _latest_tool_arguments(
    messages: list[BaseMessage], tool_name: str
) -> dict[str, Any] | None:
    for message in reversed(messages):
        if not isinstance(message, AIMessage):
            continue
        for call in reversed(message.tool_calls):
            if call.get("name") == tool_name and isinstance(call.get("args"), dict):
                return call["args"]
    return None


def _deterministic_verification_issue(
    question: str,
    task_type: str,
    candidate: str,
    messages: list[BaseMessage],
) -> str | None:
    """Reject obvious unsupported or badly formatted candidates before the LLM."""

    if not candidate:
        return "The answer candidate is empty. Continue the task and produce an answer."

    lowered_question = question.lower()
    if "just give the number" in lowered_question and not re.fullmatch(
        r"[-+]?\d+(?:\.\d+)?", candidate.strip()
    ):
        return "The question requests only a number; remove every other word and label."
    if re.search(r"\bwhat\s+(?:single\s+)?word\b", lowered_question) and len(
        re.findall(r"[\w'-]+", candidate)
    ) > 3:
        return "The question asks for a word; return only that word, not a sentence."

    if task_type in {"calculation", "mixed"} and not _successful_observations(
        messages, {"calculate", "calculate_audited_quantity"}
    ):
        return (
            "No successful calculator result supports this answer. Build the "
            "count and unit ledgers, then use the appropriate calculator tool."
        )

    if task_type in {"research", "mixed"}:
        sources = _successful_observations(messages, _FULL_SOURCE_TOOLS)
        if not sources:
            return (
                "No full source was successfully opened. Search snippets, article "
                "titles, blocked pages, and failed PDFs are not sufficient evidence."
            )
        evidence = "\n".join(text for _, text in sources).lower()
        candidate_numbers = re.findall(
            r"(?<![\w.])[-+]?\d+(?:,\d{3})*(?:\.\d+)?%?", candidate
        )
        if candidate_numbers and not all(number.lower() in evidence for number in candidate_numbers):
            calculated = "\n".join(
                text for _, text in _successful_observations(messages, {"calculate"})
            ).lower()
            if not all(number.lower() in calculated for number in candidate_numbers):
                return (
                    "The candidate number does not appear in an opened source or a "
                    "successful calculator result. Find direct supporting evidence."
                )
        if re.search(r"\bwhat\s+(?:single\s+)?word\b", lowered_question):
            word = candidate.strip(" \t\r\n\"'.,;:!?()").lower()
            if word and not re.search(rf"\b{re.escape(word)}\b", evidence):
                return (
                    "The candidate word does not occur in the opened source. Find the "
                    "exact passage instead of inferring from the title."
                )
    return None


def _verification_transcript(messages: list[BaseMessage]) -> str:
    """Give the verifier compact calls and observations without hidden reasoning."""

    blocks: list[str] = []
    for message in messages:
        if isinstance(message, AIMessage) and message.tool_calls:
            for call in message.tool_calls:
                blocks.append(
                    f"TOOL CALL {call.get('name', 'unknown')}: "
                    f"{_trace_preview(call.get('args', {}), 900)}"
                )
        elif isinstance(message, ToolMessage):
            blocks.append(
                f"TOOL RESULT {message.name or 'tool'}: "
                f"{_trace_preview(_message_text(message), 2200)}"
            )
    transcript = "\n\n".join(blocks)
    if len(transcript) > MAX_VERIFIER_EVIDENCE:
        transcript = transcript[-MAX_VERIFIER_EVIDENCE:]
    return transcript or "No tools were used."


def _tool_observation_transcript(messages: list[BaseMessage]) -> str:
    """Flatten tool results for a final model call with no structured history."""

    blocks = [
        f"{message.name or 'tool'} result:\n{_message_text(message).strip()}"
        for message in messages
        if (
            isinstance(message, ToolMessage)
            and _message_text(message).strip()
            and not _observation_failed(_message_text(message))
        )
    ]
    transcript = "\n\n".join(blocks)
    if len(transcript) > MAX_VERIFIER_EVIDENCE:
        transcript = transcript[-MAX_VERIFIER_EVIDENCE:]
    return transcript or "No tool observations are available."


def _lean_tool_round_limit(state: AgentState) -> int:
    """Give source-heavy tasks one extra action without taxing every question."""

    if state.get("task_type") in {"research", "mixed"}:
        return LEAN_RESEARCH_MAX_TOOL_ROUNDS
    return LEAN_MAX_TOOL_ROUNDS


def _visual_youtube_tool_call(state: AgentState) -> AIMessage | None:
    """Route frame-dependent YouTube questions directly to video analysis."""

    question = state.get("question", "")
    url_match = re.search(
        r"https?://(?:www\.)?(?:youtube\.com/watch\?(?:[^\s,]*&)?v=|youtu\.be/)"
        r"[A-Za-z0-9_-]{11}",
        question,
        re.IGNORECASE,
    )
    visual_cues = re.search(
        r"\b(?:on camera|simultaneously|shown|visible|visually|appears?|featured|"
        r"what (?:animal|bird|object|species)|how many .* (?:shown|visible))\b",
        question,
        re.IGNORECASE,
    )
    if not url_match or not visual_cues:
        return None
    already_called = any(
        isinstance(message, AIMessage)
        and any(
            call.get("name") == "analyze_youtube_video"
            for call in message.tool_calls
        )
        for message in state.get("messages", [])
    )
    if already_called:
        return None
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "analyze_youtube_video",
                "args": {"url_or_id": url_match.group(0), "question": question},
                "id": "lean-youtube-visual",
                "type": "tool_call",
            }
        ],
    )


def _deterministic_research_followup(state: AgentState) -> AIMessage | None:
    """Turn a useful web search into a source-opening action automatically."""

    if state.get("task_type") not in {"research", "mixed"}:
        return None
    visual_call = _visual_youtube_tool_call(state)
    if visual_call is not None:
        return visual_call
    messages = state.get("messages", [])
    if not messages or not isinstance(messages[-1], ToolMessage):
        return None
    observation = messages[-1]
    search_observation = observation if observation.name == "search_web" else None
    if observation.name in {"fetch_webpage", "fetch_pdf"} and _observation_failed(
        _message_text(observation)
    ):
        search_observation = next(
            (
                message
                for message in reversed(messages[:-1])
                if isinstance(message, ToolMessage)
                and message.name == "search_web"
                and not _observation_failed(_message_text(message))
            ),
            None,
        )
    if search_observation is None:
        return None
    text = _message_text(search_observation)
    if _observation_failed(text):
        return None
    search_args = _latest_tool_arguments(messages, "search_web") or {}
    context = f"{state.get('question', '')} {search_args.get('query', '')}"
    attempted_urls = {
        str(call.get("args", {}).get("url", ""))
        for message in messages
        if isinstance(message, AIMessage)
        for call in message.tool_calls
        if call.get("name") in {"fetch_webpage", "fetch_pdf"}
        and isinstance(call.get("args"), dict)
    }
    url = next(
        (
            candidate
            for candidate in _source_urls_from_search_result(text, context)
            if candidate not in attempted_urls
        ),
        "",
    )
    if not url:
        return None
    tool_name = "fetch_pdf" if _looks_like_pdf_url(url) else "fetch_webpage"
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": tool_name,
                "args": {"url": url},
                "id": f"lean-source-{int(state.get('tool_rounds', 0)) + 1}",
                "type": "tool_call",
            }
        ],
    )


def build_agent_graph(
    worker_models: dict[str, Runnable[Any, BaseMessage]] | None = None,
    verifier_model: Runnable[Any, BaseMessage] | None = None,
):
    """Build the planning/specialist/verification workflow.

    Models are optional so the exact topology can be rendered offline.
    """

    def planner(state: AgentState) -> dict[str, Any]:
        task_type = _classify_task(
            state.get("question", ""), bool(state.get("attachment_path"))
        )
        active_worker = {
            "attachment": "generalist",
            "calculation": "calculator",
            "general": "generalist",
            "mixed": "researcher",
            "research": "researcher",
        }[task_type]
        return {
            "task_type": task_type,
            "plan": _build_plan(task_type),
            "active_worker": active_worker,
            "verification_revisions": 0,
            "tool_rounds": 0,
            "verification_status": "pending",
            "final_answer": "",
        }

    def invoke_worker(
        state: AgentState, worker_name: str, specialist_instructions: str
    ) -> dict[str, Any]:
        literal_answer = _explicit_literal_answer(state.get("question", ""))
        if literal_answer:
            return {
                "messages": [AIMessage(content=literal_answer)],
                "active_worker": worker_name,
            }
        if worker_models is None:
            raise RuntimeError("The visualization-only graph cannot be executed.")
        model = worker_models[worker_name]
        plan = "\n".join(
            f"{number}. {step}" for number, step in enumerate(state.get("plan", []), 1)
        )
        system_prompt = (
            f"{SYSTEM_INSTRUCTIONS}\n\n{specialist_instructions}\n\n"
            f"Planner classification: {state.get('task_type', 'general')}\n"
            f"Required plan:\n{plan}"
        )
        response = model.invoke(
            [SystemMessage(content=system_prompt), *state["messages"]]
        )
        return {"messages": [response], "active_worker": worker_name}

    def tool_guard(state: AgentState) -> dict[str, int]:
        return {"tool_rounds": int(state.get("tool_rounds", 0)) + 1}

    def researcher(state: AgentState) -> dict[str, Any]:
        return invoke_worker(state, "researcher", RESEARCH_INSTRUCTIONS)

    def calculator(state: AgentState) -> dict[str, Any]:
        return invoke_worker(state, "calculator", CALCULATION_INSTRUCTIONS)

    def generalist(state: AgentState) -> dict[str, Any]:
        return invoke_worker(state, "generalist", GENERALIST_INSTRUCTIONS)

    def verifier(state: AgentState) -> dict[str, Any]:
        candidate = _latest_answer_candidate(state["messages"])
        issue = _deterministic_verification_issue(
            state.get("question", ""),
            state.get("task_type", "general"),
            candidate,
            state["messages"],
        )
        feedback = issue or ""
        approved = issue is None

        needs_model_review = (
            state.get("task_type", "general") != "calculation"
            and not _explicit_literal_answer(state.get("question", ""))
        )
        if approved and needs_model_review:
            if verifier_model is None:
                raise RuntimeError("The visualization-only graph cannot be executed.")
            review_prompt = (
                f"Original question:\n{state.get('question', '')}\n\n"
                f"Planner classification: {state.get('task_type', 'general')}\n"
                f"Plan:\n- " + "\n- ".join(state.get("plan", [])) + "\n\n"
                f"Candidate answer:\n{candidate}\n\n"
                f"Observable tool record:\n{_verification_transcript(state['messages'])}"
            )
            review = verifier_model.invoke(
                [
                    SystemMessage(content=VERIFIER_INSTRUCTIONS),
                    HumanMessage(content=review_prompt),
                ]
            )
            review_text = _message_text(review).strip()
            approved = bool(
                re.search(r"^VERDICT:\s*APPROVE\b", review_text, re.IGNORECASE)
            )
            feedback_match = re.search(
                r"^FEEDBACK:\s*(.+)$", review_text, re.IGNORECASE | re.MULTILINE
            )
            feedback = (
                feedback_match.group(1).strip()
                if feedback_match
                else "Independently recheck the reasoning and evidence."
            )

        revisions = int(state.get("verification_revisions", 0))
        if approved or revisions >= MAX_VERIFICATION_REVISIONS:
            return {
                "verification_status": "approved" if approved else "forced",
                "final_answer": candidate,
            }

        return {
            "messages": [
                HumanMessage(
                    content=(
                        "Verifier requested revision: "
                        f"{feedback} Continue from the existing evidence, use tools "
                        "as needed, and produce a corrected answer candidate."
                    )
                )
            ],
            "verification_status": "revise",
            "verification_revisions": revisions + 1,
            "final_answer": "",
        }

    def route_after_planner(state: AgentState) -> str:
        return state["active_worker"]

    def route_after_worker(state: AgentState) -> Literal["tools", "verifier"]:
        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "tools"
        return "verifier"

    def route_after_tools(state: AgentState) -> str:
        if int(state.get("tool_rounds", 0)) >= MAX_TOOL_ROUNDS:
            return "verifier"
        return state.get("active_worker", "generalist")

    def route_after_verifier(state: AgentState) -> str:
        if state.get("verification_status") == "revise":
            return state.get("active_worker", "generalist")
        return "end"

    builder = StateGraph(AgentState)
    builder.add_node("planner", planner)
    builder.add_node("researcher", researcher)
    builder.add_node("calculator", calculator)
    builder.add_node("generalist", generalist)
    builder.add_node("tools", ToolNode(ALL_TOOLS, handle_tool_errors=True))
    builder.add_node("tool_guard", tool_guard)
    builder.add_node("verifier", verifier)
    builder.add_edge(START, "planner")
    builder.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "researcher": "researcher",
            "calculator": "calculator",
            "generalist": "generalist",
        },
    )
    for worker_name in ("researcher", "calculator", "generalist"):
        builder.add_conditional_edges(
            worker_name,
            route_after_worker,
            {"tools": "tools", "verifier": "verifier"},
        )
    builder.add_edge("tools", "tool_guard")
    builder.add_conditional_edges(
        "tool_guard",
        route_after_tools,
        {
            "researcher": "researcher",
            "calculator": "calculator",
            "generalist": "generalist",
            "verifier": "verifier",
        },
    )
    builder.add_conditional_edges(
        "verifier",
        route_after_verifier,
        {
            "researcher": "researcher",
            "calculator": "calculator",
            "generalist": "generalist",
            "end": END,
        },
    )
    return builder.compile()


def build_lean_agent_graph(
    worker_models: dict[str, Runnable[Any, BaseMessage]] | None = None,
    answer_model: Runnable[Any, BaseMessage] | None = None,
):
    """Build the low-latency graph used by local Ollama by default.

    Routing is still deterministic and visible, but this profile omits the
    separate model verifier. General/calculation tasks get two tool rounds;
    research/mixed tasks get four. Exhausting that task-specific budget leads
    to a text-only final answer call with tools disabled.
    """

    def router(state: AgentState) -> dict[str, Any]:
        task_type = _classify_task(
            state.get("question", ""), bool(state.get("attachment_path"))
        )
        active_worker = {
            "attachment": "generalist",
            "calculation": "calculator",
            "general": "generalist",
            "mixed": "researcher",
            "research": "researcher",
        }[task_type]
        return {
            "task_type": task_type,
            "plan": _build_plan(task_type),
            "active_worker": active_worker,
            "tool_rounds": 0,
            "verification_status": "lean",
            "final_answer": "",
        }

    def specialist_prompt(state: AgentState) -> str:
        worker_name = state.get("active_worker", "generalist")
        instructions = {
            "researcher": RESEARCH_INSTRUCTIONS,
            "calculator": CALCULATION_INSTRUCTIONS,
            "generalist": GENERALIST_INSTRUCTIONS,
        }[worker_name]
        plan = "\n".join(
            f"{number}. {step}"
            for number, step in enumerate(state.get("plan", []), 1)
        )
        return (
            f"{SYSTEM_INSTRUCTIONS}\n\n{LEAN_INSTRUCTIONS}\n\n{instructions}\n\n"
            f"Deterministic classification: {state.get('task_type', 'general')}\n"
            f"Tool-round limit for this task: {_lean_tool_round_limit(state)}\n"
            f"Checklist:\n{plan}"
        )

    def assistant(state: AgentState) -> dict[str, Any]:
        literal_answer = _explicit_literal_answer(state.get("question", ""))
        if literal_answer:
            return {"messages": [AIMessage(content=literal_answer)]}
        if worker_models is None:
            raise RuntimeError("The visualization-only graph cannot be executed.")
        deterministic_followup = _deterministic_research_followup(state)
        if deterministic_followup is not None:
            return {"messages": [deterministic_followup]}
        worker_name = state.get("active_worker", "generalist")
        response = worker_models[worker_name].invoke(
            [SystemMessage(content=specialist_prompt(state)), *state["messages"]]
        )
        return {"messages": [response]}

    def tool_guard(state: AgentState) -> dict[str, int]:
        return {"tool_rounds": int(state.get("tool_rounds", 0)) + 1}

    def forced_answer(state: AgentState) -> dict[str, Any]:
        if answer_model is None:
            raise RuntimeError("The visualization-only graph cannot be executed.")
        final_prompt = (
            f"Original question:\n{state.get('question', '')}\n\n"
            "Available tool observations:\n"
            f"{_tool_observation_transcript(state['messages'])}\n\n"
            "Return only the exact answer requested by the original question."
        )
        response = answer_model.invoke(
            [
                SystemMessage(
                    content=(
                        "You are the final-answer stage. Tools and function calls "
                        "are unavailable. Use only the supplied question and "
                        "plain-text observations. Never output a tool request, "
                        "function-call syntax, reasoning, citations, or labels."
                    )
                ),
                HumanMessage(content=final_prompt),
            ]
        )
        return {"messages": [response]}

    def finalize(state: AgentState) -> dict[str, str]:
        return {
            "final_answer": _latest_answer_candidate(state["messages"]),
            "verification_status": "lean",
        }

    def route_after_assistant(state: AgentState) -> Literal["tools", "finalize"]:
        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "tools"
        return "finalize"

    def route_after_tool_guard(
        state: AgentState,
    ) -> Literal["assistant", "forced_answer"]:
        if int(state.get("tool_rounds", 0)) >= _lean_tool_round_limit(state):
            return "forced_answer"
        return "assistant"

    builder = StateGraph(AgentState)
    builder.add_node("router", router)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(ALL_TOOLS, handle_tool_errors=True))
    builder.add_node("tool_guard", tool_guard)
    builder.add_node("forced_answer", forced_answer)
    builder.add_node("finalize", finalize)
    builder.add_edge(START, "router")
    builder.add_edge("router", "assistant")
    builder.add_conditional_edges(
        "assistant",
        route_after_assistant,
        {"tools": "tools", "finalize": "finalize"},
    )
    builder.add_edge("tools", "tool_guard")
    builder.add_conditional_edges(
        "tool_guard",
        route_after_tool_guard,
        {"assistant": "assistant", "forced_answer": "forced_answer"},
    )
    builder.add_edge("forced_answer", "finalize")
    builder.add_edge("finalize", END)
    return builder.compile()


def _graph_for_profile(profile: str | None = None):
    selected = (profile or _selected_agent_profile()).strip().lower()
    selected = {"fast": "lean", "full": "robust"}.get(selected, selected)
    if selected == "lean":
        return build_lean_agent_graph()
    if selected == "robust":
        return build_agent_graph()
    raise AgentConfigurationError(
        "AGENT_PROFILE must be either 'lean' or 'robust'."
    )


def agent_graph_mermaid(profile: str | None = None) -> str:
    """Return Mermaid source for the selected LangGraph profile."""

    return _graph_for_profile(profile).get_graph().draw_mermaid()


def render_agent_graph_png(profile: str | None = None) -> bytes:
    """Render the selected workflow through LangGraph's Mermaid renderer."""

    return _graph_for_profile(profile).get_graph().draw_mermaid_png(
        background_color="white",
        max_retries=3,
        retry_delay=2.0,
    )


def _message_text(message: BaseMessage) -> str:
    """Extract final text from either string or structured message content."""

    content = message.content
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and isinstance(block.get("text"), str):
            parts.append(block["text"])
    return "\n".join(parts)


def _trace_preview(value: Any, limit: int = TRACE_PREVIEW_LIMIT) -> str:
    """Make tool arguments and observations compact enough for a debug trace."""

    if isinstance(value, str):
        text = value
    else:
        import json

        text = json.dumps(value, ensure_ascii=False, default=str)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit] + "... [trace truncated]"


def build_execution_trace(messages: list[BaseMessage]) -> list[dict[str, str]]:
    """Summarize actions and observations without exposing private reasoning."""

    trace: list[dict[str, str]] = []
    for message in messages:
        if isinstance(message, AIMessage) and message.tool_calls:
            for call in message.tool_calls:
                trace.append(
                    {
                        "type": "tool_call",
                        "tool": str(call.get("name", "unknown")),
                        "detail": _trace_preview(call.get("args", {})),
                    }
                )
        elif isinstance(message, ToolMessage):
            trace.append(
                {
                    "type": "tool_result",
                    "tool": str(message.name or "tool"),
                    "detail": _trace_preview(_message_text(message)),
                }
            )
        elif isinstance(message, AIMessage):
            text = _message_text(message)
            trace.append(
                {
                    "type": "answer_candidate",
                    "tool": "assistant",
                    "detail": _trace_preview(text) if text else "[empty response]",
                }
            )
        elif isinstance(message, HumanMessage):
            text = _message_text(message)
            prefix = "Verifier requested revision:"
            if text.startswith(prefix):
                trace.append(
                    {
                        "type": "verification",
                        "tool": "verifier",
                        "detail": _trace_preview(text.removeprefix(prefix).strip()),
                    }
                )
    return trace


def format_execution_trace(trace: list[dict[str, str]]) -> str:
    """Format a compact trace for the terminal and Gradio practice tab."""

    if not trace:
        return "No tool calls or completed model response were recorded."
    labels = {
        "profile": "PROFILE",
        "plan": "PLAN",
        "model_fallback": "MODEL FALLBACK",
        "tool_call": "TOOL CALL",
        "tool_result": "TOOL RESULT",
        "answer_candidate": "ANSWER CANDIDATE",
        "verification": "VERIFIER",
        "final_answer": "FINAL RESPONSE",
    }
    lines = []
    for number, item in enumerate(trace, start=1):
        label = labels.get(item["type"], item["type"].upper())
        lines.append(
            f"{number}. {label} - {item['tool']}\n   {item['detail']}"
        )
    return "\n".join(lines)


class GaiaAgent:
    """LangGraph tool-calling agent with the assignment's call signature."""

    def __init__(self) -> None:
        self.last_trace: list[dict[str, str]] = []
        self.backend = _selected_backend()
        self.profile = _selected_agent_profile(self.backend)
        self.fallback_model_id = ""
        fallback_model = None
        if self.backend == "huggingface":
            token = os.getenv("HF_TOKEN")
            if not _token_is_configured(token):
                raise AgentConfigurationError(
                    "HF_TOKEN is not set. Add your Hugging Face read token to "
                    ".env locally or to your Space repository secrets."
                )

            base_model_id = os.getenv("GAIA_MODEL", DEFAULT_MODEL)
            provider = os.getenv("HF_INFERENCE_PROVIDER") or None
            self.model_id = _routed_model_id(base_model_id, provider)
            self.provider = provider or "auto"
            self.base_url = HF_ROUTER_URL
            model = ChatOpenAI(
                model=self.model_id,
                api_key=token,
                base_url=self.base_url,
                temperature=0.0,
                max_tokens=1200,
                timeout=120,
                max_retries=2,
            )
        elif self.backend == "ollama":
            try:
                from langchain_ollama import ChatOllama
            except ImportError as exc:
                raise AgentConfigurationError(
                    "The Ollama backend needs langchain-ollama. Run: "
                    "python -m pip install -r requirements.txt"
                ) from exc

            self.model_id = os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
            self.base_url = os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL)
            self.provider = "local Ollama"
            reasoning = _env_bool("OLLAMA_REASONING", False)
            context_length = _env_positive_int("OLLAMA_CONTEXT_LENGTH", 8192)
            num_predict = _env_positive_int("OLLAMA_NUM_PREDICT", 600)
            try:
                model = ChatOllama(
                    model=self.model_id,
                    base_url=self.base_url,
                    temperature=0.0,
                    num_ctx=context_length,
                    num_predict=num_predict,
                    reasoning=reasoning,
                    validate_model_on_init=True,
                )
            except Exception as exc:
                raise AgentConfigurationError(
                    f"Could not start local model '{self.model_id}' at "
                    f"{self.base_url}. Make sure Ollama is running, then run: "
                    f"ollama pull {self.model_id}"
                ) from exc
        else:
            api_key = os.getenv("GEMINI_API_KEY")
            if not _gemini_key_is_configured(api_key):
                raise AgentConfigurationError(
                    "GEMINI_API_KEY is not set. Add it to .env locally or to "
                    "your Space repository secrets."
                )
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
            except ImportError as exc:
                raise AgentConfigurationError(
                    "The Gemini backend needs langchain-google-genai. Run: "
                    "python -m pip install -r requirements.txt"
                ) from exc

            self.model_id = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
            self.fallback_model_id = os.getenv(
                "GEMINI_FALLBACK_MODEL", DEFAULT_GEMINI_FALLBACK_MODEL
            ).strip()
            if self.fallback_model_id == self.model_id:
                self.fallback_model_id = ""
            self.provider = "Google Gemini Developer API"
            self.base_url = "https://generativelanguage.googleapis.com"
            thinking_level = _gemini_thinking_level()
            max_tokens = _env_positive_int("GEMINI_MAX_OUTPUT_TOKENS", 1200)
            try:
                model = ChatGoogleGenerativeAI(
                    model=self.model_id,
                    api_key=api_key,
                    **_gemini_sampling_kwargs(self.model_id),
                    thinking_level=thinking_level,
                    max_tokens=max_tokens,
                    timeout=120,
                    max_retries=2,
                )
                if self.fallback_model_id:
                    fallback_model = ChatGoogleGenerativeAI(
                        model=self.fallback_model_id,
                        api_key=api_key,
                        **_gemini_sampling_kwargs(self.fallback_model_id),
                        thinking_level=thinking_level,
                        max_tokens=max_tokens,
                        timeout=120,
                        max_retries=2,
                    )
            except Exception as exc:
                raise AgentConfigurationError(
                    f"Could not initialize Gemini model '{self.model_id}'. "
                    "Check GEMINI_API_KEY, GEMINI_MODEL, and the installed "
                    "langchain-google-genai version."
                ) from exc

        self.graph, self.model_with_tools = self._compile_graph(model)
        self.fallback_graph = None
        if fallback_model is not None:
            self.fallback_graph, _ = self._compile_graph(fallback_model)

    def _compile_graph(self, model: Any) -> tuple[Runnable, Any]:
        """Compile the selected profile around one model instance."""

        model_with_tools = model.bind_tools(ALL_TOOLS)
        worker_models = {
            "researcher": model.bind_tools(RESEARCH_TOOLS),
            "calculator": model.bind_tools(CALCULATION_TOOLS),
            "generalist": model_with_tools,
        }
        if self.profile == "lean":
            graph = build_lean_agent_graph(worker_models, model)
            self.graph_step_limit = LEAN_MAX_GRAPH_STEPS
        else:
            graph = build_agent_graph(worker_models, model)
            self.graph_step_limit = MAX_GRAPH_STEPS
        return graph, model_with_tools

    def _raise_run_error(self, exc: Exception) -> None:
        """Translate common provider failures into concise learner guidance."""

        detail = str(exc)
        if self.backend == "huggingface" and "402" in detail:
            raise AgentConfigurationError(
                "Hugging Face rejected the request because the account has "
                "no remaining inference credits. Add credits, wait for the "
                "monthly reset, or use MODEL_BACKEND=ollama locally."
            ) from exc
        if self.backend == "huggingface" and (
            "model_not_supported" in detail
            or "not supported by any provider" in detail
        ):
            raise AgentConfigurationError(
                f"Model '{self.model_id}' is not available through provider "
                f"'{self.provider}'. Change GAIA_MODEL or "
                "HF_INFERENCE_PROVIDER in .env."
            ) from exc
        if self.backend == "gemini" and any(
            marker in detail.lower()
            for marker in (
                "api_key_invalid",
                "api key not valid",
                "permission_denied",
                "unauthenticated",
            )
        ):
            raise AgentConfigurationError(
                "Gemini rejected the API key. Replace GEMINI_API_KEY in "
                ".env or in your Space repository secrets."
            ) from exc
        if self.backend == "gemini" and any(
            marker in detail.lower()
            for marker in ("resource_exhausted", "quota", "429")
        ):
            raise AgentConfigurationError(
                "Gemini quota was exhausted or rate-limited. Wait for the "
                "quota window to reset, reduce requests, or enable billing."
            ) from exc
        if self.backend == "gemini" and _is_transient_gemini_error(exc):
            models = self.model_id
            if self.fallback_model_id:
                models += f" and fallback {self.fallback_model_id}"
            raise AgentConfigurationError(
                f"Google temporarily could not serve {models} after automatic "
                "retries. This is a provider capacity error, not a bad API key. "
                "Wait briefly and run the question again."
            ) from exc
        if self.backend == "ollama" and any(
            marker in detail.lower()
            for marker in ("connection refused", "connecterror", "failed to connect")
        ):
            raise AgentConfigurationError(
                f"Could not reach Ollama at {self.base_url}. Start the Ollama "
                "application or run 'ollama serve'."
            ) from exc
        if "recursion limit" in detail.lower():
            raise RuntimeError(
                "The LangGraph agent reached its step limit before producing "
                f"an answer while using the {self.profile!r} profile. Improve "
                "the prompt/tools or choose the other AGENT_PROFILE."
            ) from exc
        raise exc

    def __call__(
        self,
        question: str,
        task_id: str = "practice",
        attachment_path: Path | None = None,
    ) -> str:
        """Solve one question and return only its normalized final answer."""

        self.last_trace = []
        reset_research_session()
        set_task_question(question)

        attachment_note = "There is no attachment for this task."
        if attachment_path is not None:
            attachment_note = (
                "An attachment is available at this exact local path: "
                f"{attachment_path}. Inspect it with a file tool before answering."
            )

        prompt = (
            f"Task ID: {task_id}\n"
            f"{attachment_note}\n\n"
            f"Question:\n{question}\n\n"
            "Return only the final answer requested by the question."
        )
        graph_input = {
            "messages": [HumanMessage(content=prompt)],
            "question": question,
            "task_id": task_id,
            "attachment_path": str(attachment_path or ""),
        }
        used_fallback = False
        try:
            result = self.graph.invoke(
                graph_input,
                config={"recursion_limit": self.graph_step_limit},
            )
        except Exception as exc:  # noqa: BLE001 - providers use varied exception types.
            if (
                self.backend == "gemini"
                and self.fallback_graph is not None
                and _is_retryable_gemini_error(exc)
            ):
                reset_research_session()
                set_task_question(question)
                try:
                    result = self.fallback_graph.invoke(
                        graph_input,
                        config={"recursion_limit": self.graph_step_limit},
                    )
                    used_fallback = True
                except Exception as fallback_exc:  # noqa: BLE001 - same provider boundary.
                    self._raise_run_error(fallback_exc)
            else:
                self._raise_run_error(exc)

        messages = result.get("messages", [])
        plan = result.get("plan", [])
        task_type = str(result.get("task_type", "general"))
        self.last_trace = [
            {
                "type": "profile",
                "tool": self.profile,
                "detail": (
                    "low-latency tool loop"
                    if self.profile == "lean"
                    else "planner, specialists, and verifier"
                ),
            },
            {
                "type": "plan",
                "tool": task_type,
                "detail": _trace_preview(plan),
            },
            *(
                [
                    {
                        "type": "model_fallback",
                        "tool": self.fallback_model_id,
                        "detail": (
                            f"Primary model {self.model_id} was temporarily "
                            "unavailable; restarted this question with the fallback."
                        ),
                    }
                ]
                if used_fallback
                else []
            ),
            *build_execution_trace(messages),
        ]
        if not messages:
            raise RuntimeError("The LangGraph run returned no messages.")
        evidence_answer = _quoted_answer_from_evidence(question, messages)
        if evidence_answer:
            self.last_trace.append(
                {
                    "type": "evidence_extraction",
                    "tool": "quoted-word matcher",
                    "detail": _trace_preview(evidence_answer),
                }
            )
        answer = normalize_answer_for_question(
            question, evidence_answer or str(result.get("final_answer", ""))
        )
        if not answer:
            raise RuntimeError(
                "The agent returned an empty answer. The model did not finish "
                f"within the {self.profile!r} profile's tool budget; inspect "
                "--trace or try a stronger model."
            )
        self.last_trace.append(
            {
                "type": "final_answer",
                "tool": str(result.get("verification_status", "verifier")),
                "detail": _trace_preview(answer),
            }
        )
        return answer

    def trace_text(self) -> str:
        """Return the most recent run's compact, reasoning-free trace."""

        return format_execution_trace(self.last_trace)
