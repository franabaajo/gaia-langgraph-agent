"""Estimate agent quality on GAIA Level 1 without submitting to a leaderboard.

The runner reads validation metadata already present in the example-project
folder, never sends reference answers to the agent, and skips questions whose
attachments are not available. Its score is a development proxy, not the
official course score.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from agent import GaiaAgent, configuration_report

PROJECT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = PROJECT_DIR.parent
DEFAULT_METADATA_CANDIDATES = (
    WORKSPACE_DIR / "LEADERBOARD EXAMPLES" / "gaia-agent" / "metadata.jsonl",
    WORKSPACE_DIR
    / "LEADERBOARD EXAMPLES"
    / "Final_Project_Cource"
    / "metadata.jsonl",
)


def answers_match(submitted: str, expected: str) -> bool:
    """Apply the course's case-insensitive, outer-whitespace exact match."""

    return submitted.strip().lower() == expected.strip().lower()


def _is_rate_limit_error(exc: Exception) -> bool:
    detail = str(exc).lower()
    return any(
        marker in detail
        for marker in ("429", "resource_exhausted", "quota", "rate limit", "rate-limit")
    )


def _answer_with_rate_limit_retries(
    agent: GaiaAgent,
    *,
    question: str,
    task_id: str,
    retries: int,
    base_delay: float,
) -> str:
    """Retry only provider throttling; surface unrelated failures immediately."""

    for attempt in range(retries + 1):
        try:
            return agent(question=question, task_id=task_id)
        except Exception as exc:
            if not _is_rate_limit_error(exc) or attempt >= retries:
                raise
            delay = base_delay * (2**attempt)
            print(
                f"  Rate limited; retrying this question in {delay:g}s "
                f"({attempt + 1}/{retries})."
            )
            time.sleep(delay)
    raise RuntimeError("Unreachable retry state.")


def _default_metadata_path() -> Path:
    for candidate in DEFAULT_METADATA_CANDIDATES:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Could not find metadata.jsonl in either example project. Pass its "
        "location with --metadata."
    )


def _load_level_one_without_attachments(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            item = json.loads(line)
            if int(item.get("Level", 0)) != 1:
                continue
            if item.get("file_name"):
                continue
            if not item.get("Question") or not item.get("Final answer"):
                continue
            rows.append(item)
    if not rows:
        raise RuntimeError("No answerable Level-1 rows were found in the metadata.")
    return rows


def _select_questions(
    questions: list[dict[str, Any]],
    *,
    task_id: str,
    seed: int,
    limit: int,
) -> list[dict[str, Any]]:
    """Select one task by ID or the usual deterministic shuffled sample."""

    if task_id:
        return [item for item in questions if str(item.get("task_id")) == task_id]
    selected = list(questions)
    random.Random(seed).shuffle(selected)
    return selected[:limit] if limit else selected


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata",
        type=Path,
        help="Path to GAIA validation metadata.jsonl (uses the example copy by default)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of questions to run; use 0 for every eligible question",
    )
    parser.add_argument(
        "--task-id",
        default="",
        help="Run exactly one eligible metadata task by ID; overrides --limit and --seed",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed so repeated runs select the same questions",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("local_eval_results.json"),
        help="Where to save predictions and pass/fail results",
    )
    parser.add_argument(
        "--rate-limit-retries",
        type=int,
        default=2,
        help="Retries per question after a 429/quota response (default: 2)",
    )
    parser.add_argument(
        "--rate-limit-delay",
        type=float,
        default=20.0,
        help="Initial rate-limit retry delay in seconds; doubles per retry",
    )
    parser.add_argument(
        "--request-delay",
        type=float,
        default=0.0,
        help="Optional pause between evaluation questions",
    )
    args = parser.parse_args()

    if args.limit < 0:
        parser.error("--limit cannot be negative")
    if args.rate_limit_retries < 0:
        parser.error("--rate-limit-retries cannot be negative")
    if args.rate_limit_delay < 0 or args.request_delay < 0:
        parser.error("evaluation delays cannot be negative")

    metadata_path = (args.metadata or _default_metadata_path()).resolve()
    if not metadata_path.is_file():
        parser.error(f"Metadata file does not exist: {metadata_path}")

    questions = _select_questions(
        _load_level_one_without_attachments(metadata_path),
        task_id=args.task_id.strip(),
        seed=args.seed,
        limit=args.limit,
    )
    if args.task_id and not questions:
        parser.error(
            f"Task ID is not an eligible Level-1 no-attachment question: {args.task_id}"
        )

    config_report = configuration_report()
    print("Local GAIA proxy evaluation")
    print(f"Backend: {next(line for line in config_report if 'MODEL_BACKEND=' in line)}")
    print(f"Profile: {next(line for line in config_report if 'AGENT_PROFILE=' in line)}")
    print(f"Metadata: {metadata_path}")
    print(f"Questions: {len(questions)} (Level 1, no attachment)")
    print("Reference answers are used only for scoring, never added to the prompt.\n")

    agent = GaiaAgent()
    correct = 0
    results: list[dict[str, Any]] = []
    for index, item in enumerate(questions, start=1):
        task_id = str(item["task_id"])
        try:
            answer = _answer_with_rate_limit_retries(
                agent,
                question=str(item["Question"]),
                task_id=f"local-eval-{task_id}",
                retries=args.rate_limit_retries,
                base_delay=args.rate_limit_delay,
            )
            passed = answers_match(answer, str(item["Final answer"]))
            error = ""
        except Exception as exc:  # noqa: BLE001 - record one failure and keep scoring.
            answer = ""
            passed = False
            error = str(exc)
        trace = list(agent.last_trace)

        correct += int(passed)
        status = "PASS" if passed else "FAIL"
        print(f"[{index:02d}/{len(questions):02d}] {status}  {task_id}")
        if answer:
            print(f"  Agent answer: {answer}")
        if error:
            print(f"  Error: {error}")
        if not passed and trace:
            print("  Execution trace:")
            print("\n".join(f"    {line}" for line in agent.trace_text().splitlines()))
        results.append(
            {
                "task_id": task_id,
                "submitted_answer": answer,
                "correct": passed,
                "error": error,
                "trace": trace,
            }
        )
        if args.request_delay and index < len(questions):
            time.sleep(args.request_delay)

    total = len(questions)
    score = 100.0 * correct / total if total else 0.0
    output = {
        "kind": "unofficial_local_proxy",
        "level": 1,
        "attachments_included": False,
        "seed": args.seed,
        "correct": correct,
        "total": total,
        "score_percent": round(score, 2),
        "results": results,
    }
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print(f"\nProxy exact-match score: {score:.1f}% ({correct}/{total})")
    print(f"Results saved to: {args.output.resolve()}")
    print(
        "This is not the official course score: it samples the broader GAIA "
        "validation set and excludes attachment questions."
    )


if __name__ == "__main__":
    main()
