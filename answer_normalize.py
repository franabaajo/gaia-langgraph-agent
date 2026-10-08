"""Conservative final-answer cleanup for an exact-match benchmark."""

from __future__ import annotations

import re

_ANSWER_PREFIX = re.compile(r"^(?:final\s+answer|answer)\s*:\s*", re.IGNORECASE)
_LABELED_ANSWER = re.compile(
    r"(?:^|\n)\s*(?:final\s+answer|answer)\s*:\s*([^\r\n]+)",
    re.IGNORECASE,
)
_NUMBER_WITH_UNIT = re.compile(
    r"^(?P<number>[-+]?(?:(?:\d{1,3}(?:,\d{3})+)|\d+)(?:\.\d+)?"
    r"(?:[eE][-+]?\d+)?)"
    r"(?:\s+(?P<word_unit>[A-Za-zµμ°%][A-Za-z0-9µμ°%/^·⋅*()._-]*"
    r"(?:\s+[A-Za-zµμ°%][A-Za-z0-9µμ°%/^·⋅*()._-]*){0,2})"
    r"|(?P<symbol_unit>[%°]))$"
)
_MEASURED_QUANTITY = (
    r"volume|mass|weight|length|distance|height|area|speed|rate|temperature|"
    r"duration|energy|power|pressure|density|angle|frequency|amount|measurement"
)
_CONCLUSION_COUNT = re.compile(
    r"\b(?:therefore|thus|hence|so|the\s+(?:final\s+)?answer\s+is)"
    r"\s*[:,]?\s*\**([-+]?(?:(?:\d{1,3}(?:,\d{3})+)|\d+))\**\b",
    re.IGNORECASE,
)


def normalize_answer(value: object) -> str:
    """Remove common model wrappers without changing answer punctuation.

    Do not aggressively reformat answers here: ``$123.00`` and ``123`` may be
    meaningfully different to an exact-match evaluator.
    """

    answer = str(value).strip()

    # Remove a complete Markdown fence, but preserve any content within it.
    if answer.startswith("```") and answer.endswith("```"):
        lines = answer.splitlines()
        if len(lines) >= 3:
            answer = "\n".join(lines[1:-1]).strip()

    labeled = _LABELED_ANSWER.findall(answer)
    if labeled:
        answer = labeled[-1].strip()
    answer = _ANSWER_PREFIX.sub("", answer, count=1).strip()

    # Exact-match graders treat decorative Markdown as extra answer text.
    for wrapper in ("**", "__", "*", "_", "`"):
        if (
            answer.startswith(wrapper)
            and answer.endswith(wrapper)
            and len(answer) > 2 * len(wrapper)
            and "\n" not in answer
        ):
            answer = answer[len(wrapper) : -len(wrapper)].strip()
            break
    return answer


def _normalize_unit_notation(value: str) -> str:
    """Make common superscript spellings comparable without changing output."""

    return " ".join(
        value.lower().translate(str.maketrans({"²": "^2", "³": "^3"})).split()
    )


def normalize_answer_for_question(question: str, value: object) -> str:
    """Remove a redundant measurement unit only when the request supplies it.

    This is intentionally narrower than general unit stripping. It applies only
    to a single numeric answer followed by a unit, requires that same unit in an
    output-unit phrase such as ``volume in m^3``, and preserves the unit whenever
    the question explicitly asks for it. It does not use reference answers.
    """

    answer = normalize_answer(value)
    normalized_question = _normalize_unit_notation(question)

    if re.search(r"^\s*who\b", normalized_question):
        labeled_person = re.fullmatch(
            r"(?:user(?:name)?|nominator)\s*:\s*(.+)",
            answer,
            re.IGNORECASE,
        )
        if labeled_person:
            answer = labeled_person.group(1).strip()

    if (
        re.search(r"\b(?:without|no)\s+abbreviations?\b", normalized_question)
        and re.search(r"\b(?:city|location|place)\b", normalized_question)
    ):
        expansions = {
            r"\bSt\.\s*": "Saint ",
            r"\bMt\.\s*": "Mount ",
            r"\bFt\.\s*": "Fort ",
        }
        for pattern, replacement in expansions.items():
            answer = re.sub(pattern, replacement, answer, flags=re.IGNORECASE)

    # Small/hosted models occasionally append the requested answer as a final
    # standalone line after an explanation. Recover only a clearly numeric tail
    # when the question itself requests a number or measured quantity.
    expects_number = bool(
        re.search(
            rf"\b(?:how many|how much|number of|{_MEASURED_QUANTITY})\b",
            normalized_question,
        )
    )
    if expects_number and "\n" in answer:
        final_line = answer.splitlines()[-1].strip()
        if re.fullmatch(
            r"[-+]?(?:(?:\d{1,3}(?:,\d{3})+)|\d+)(?:\.\d+)?"
            r"(?:[eE][-+]?\d+)?(?:\s+[A-Za-z\u00b5\u03bc\u00b0%]"
            r"[A-Za-z0-9\u00b5\u03bc\u00b0%/^\u00b7\u22c5*()._-]*)?",
            final_line,
        ):
            answer = final_line

    # Count questions require one number. Recover it only from an explicit
    # conclusion cue, avoiding unrelated dates and list item numbers earlier
    # in an explanation.
    if re.search(r"\b(?:how many|number of)\b", normalized_question):
        conclusion_counts = _CONCLUSION_COUNT.findall(answer)
        if conclusion_counts:
            answer = conclusion_counts[-1]

    match = _NUMBER_WITH_UNIT.fullmatch(answer)
    if not match:
        return answer

    unit = match.group("word_unit") or match.group("symbol_unit") or ""
    normalized_unit = _normalize_unit_notation(unit)
    if normalized_unit in {"am", "pm", "utc", "gmt"}:
        return answer

    unit_pattern = re.escape(normalized_unit).replace(r"\ ", r"\s+")
    unit_end = r"(?=$|\s|[?.,;:)])"

    explicitly_requests_unit = any(
        re.search(pattern, normalized_question, re.IGNORECASE)
        for pattern in (
            r"\b(?:include|retain|show|state|write)\b[^?.\n]{0,60}\bunits?\b",
            r"\bwith\s+(?:the\s+)?units?\b",
            (
                rf"\b(?:include|retain|show|state|write)\b[^?.\n]{{0,60}}"
                rf"{unit_pattern}{unit_end}"
            ),
        )
    )
    if explicitly_requests_unit:
        return answer

    requested_in_that_unit = any(
        re.search(pattern, normalized_question, re.IGNORECASE)
        for pattern in (
            (
                rf"\b(?:{_MEASURED_QUANTITY})\s+(?:measured\s+)?in\s+"
                rf"(?:the\s+)?{unit_pattern}{unit_end}"
            ),
            (
                rf"\b(?:answer|result|value)\s+in\s+(?:the\s+)?"
                rf"{unit_pattern}{unit_end}"
            ),
            (
                rf"\b(?:express|report|provide|give|return)\b[^?.\n]{{0,80}}"
                rf"\bin\s+(?:the\s+)?{unit_pattern}{unit_end}"
            ),
        )
    )
    if requested_in_that_unit:
        return match.group("number")
    return answer
