"""Run the official course questions locally and explicitly submit saved answers."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

from agent import AgentConfigurationError, GaiaAgent, configuration_report
from answer_normalize import normalize_answer_for_question

DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"
DEFAULT_OUTPUT = Path("official_course_results.json")
REQUEST_TIMEOUT = 30
GAIA_DATASET_ID = "gaia-benchmark/GAIA"
GAIA_VALIDATION_DIR = "2023/validation"
SPACE_ID_PATTERN = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$"
)


def fetch_official_questions(api_url: str) -> list[dict[str, Any]]:
    """Fetch the current official evaluation set without reference answers."""

    response = requests.get(f"{api_url}/questions", timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    questions = response.json()
    if not isinstance(questions, list) or not questions:
        raise RuntimeError("The official questions endpoint returned no questions.")
    return questions


def download_attachment(
    item: dict[str, Any], destination: Path, api_url: str
) -> Path | None:
    """Download an attachment, falling back to the gated GAIA dataset."""

    file_name = item.get("file_name")
    task_id = item.get("task_id")
    if not file_name or not task_id:
        return None
    safe_name = Path(str(file_name)).name
    if safe_name != str(file_name) or safe_name in {"", ".", ".."}:
        raise ValueError(f"Unsafe official attachment filename: {file_name!r}")

    output_path = destination / f"{task_id}_{safe_name}"
    response = requests.get(
        f"{api_url}/files/{task_id}",
        timeout=REQUEST_TIMEOUT,
    )
    if response.status_code != 404:
        response.raise_for_status()
        output_path.write_bytes(response.content)
        return output_path

    print(
        "  Official attachment endpoint returned 404; "
        "trying the gated GAIA dataset."
    )
    try:
        cached_path = hf_hub_download(
            repo_id=GAIA_DATASET_ID,
            filename=f"{GAIA_VALIDATION_DIR}/{safe_name}",
            repo_type="dataset",
        )
    except Exception as exc:
        raise RuntimeError(
            "The official attachment endpoint returned 404 and the GAIA "
            "fallback could not download the file. Accept the dataset terms "
            "at https://huggingface.co/datasets/gaia-benchmark/GAIA and run "
            "'hf auth login' with a read-enabled token."
        ) from exc
    return Path(cached_path)


def _is_rate_limit_error(exc: Exception) -> bool:
    detail = str(exc).lower()
    return any(
        marker in detail
        for marker in ("429", "resource_exhausted", "quota", "rate limit", "rate-limit")
    )


def _answer_with_retries(
    agent: GaiaAgent,
    *,
    question: str,
    task_id: str,
    attachment_path: Path | None,
    retries: int,
    base_delay: float,
) -> str:
    for attempt in range(retries + 1):
        try:
            return agent(
                question=question,
                task_id=task_id,
                attachment_path=attachment_path,
            )
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


def _load_saved_results(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("kind") != "official_course_local_run":
        raise ValueError(f"{path} is not an official course results file.")
    return {
        str(item["task_id"]): item
        for item in data.get("results", [])
        if item.get("task_id")
    }


def _save_results(
    path: Path,
    questions: list[dict[str, Any]],
    saved: dict[str, dict[str, Any]],
) -> None:
    ordered = [
        saved[str(item["task_id"])]
        for item in questions
        if str(item.get("task_id", "")) in saved
    ]
    ready = sum(
        bool(item.get("submitted_answer")) and not item.get("error") for item in ordered
    )
    output = {
        "kind": "official_course_local_run",
        "answers_ready": ready,
        "official_question_count": len(questions),
        "results": ordered,
    }
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")


def _successful_answers(path: Path) -> dict[str, str]:
    return {
        task_id: str(item["submitted_answer"])
        for task_id, item in _load_saved_results(path).items()
        if item.get("submitted_answer") and not item.get("error")
    }


def build_submission_payload(
    *,
    username: str,
    agent_space: str,
    questions: list[dict[str, Any]],
    saved_answers: dict[str, str],
) -> dict[str, Any]:
    """Build the exact payload documented by the official course template."""

    username = username.strip()
    agent_space = agent_space.strip().strip("/")
    if not username:
        raise ValueError("HF_USERNAME is missing from .env.")
    if not SPACE_ID_PATTERN.fullmatch(agent_space):
        raise ValueError("HF_AGENT_SPACE must use the form username/space-name.")
    if agent_space.split("/", 1)[0].casefold() != username.casefold():
        raise ValueError("HF_USERNAME must match the owner in HF_AGENT_SPACE.")

    answers = [
        {
            "task_id": str(item.get("task_id", "")),
            "submitted_answer": normalize_answer_for_question(
                str(item.get("question", "")),
                saved_answers[str(item.get("task_id", ""))],
            ),
        }
        for item in questions
        if saved_answers.get(str(item.get("task_id", "")))
    ]
    if not answers:
        raise ValueError(
            "No successful official answers are saved. Run: "
            "python submit_results.py --all"
        )

    return {
        "username": username,
        "agent_code": (
            f"https://huggingface.co/spaces/{agent_space}/tree/main"
        ),
        "answers": answers,
    }


def run_questions(args: argparse.Namespace, api_url: str) -> int:
    questions = fetch_official_questions(api_url)
    if args.task_id:
        selected = [
            item
            for item in questions
            if str(item.get("task_id", "")) == args.task_id
        ]
        if not selected:
            raise ValueError(f"Unknown official task ID: {args.task_id}")
    elif args.all:
        selected = questions
    else:
        selected = questions[: args.limit]

    report = configuration_report()
    print("Official GAIA course evaluation (local inference)")
    print(f"Backend: {next(line for line in report if 'MODEL_BACKEND=' in line)}")
    print(f"Profile: {next(line for line in report if 'AGENT_PROFILE=' in line)}")
    print(f"Questions selected: {len(selected)} of {len(questions)}")
    print(f"Results file: {args.output.resolve()}")
    print("The official API supplies questions, not reference answers.\n")

    saved = _load_saved_results(args.output)
    agent = GaiaAgent()
    with tempfile.TemporaryDirectory(prefix="gaia_attachments_official_") as temp_dir:
        attachment_dir = Path(temp_dir)
        for index, item in enumerate(selected, start=1):
            task_id = str(item.get("task_id", ""))
            question = item.get("question")
            if not task_id or not isinstance(question, str):
                print(f"[{index:02d}/{len(selected):02d}] SKIP malformed question")
                continue

            previous = saved.get(task_id, {})
            if (
                not args.force
                and previous.get("submitted_answer")
                and not previous.get("error")
            ):
                print(f"[{index:02d}/{len(selected):02d}] CACHED {task_id}")
                continue

            agent.last_trace = []
            try:
                attachment_path = download_attachment(item, attachment_dir, api_url)
                answer = _answer_with_retries(
                    agent,
                    question=question,
                    task_id=task_id,
                    attachment_path=attachment_path,
                    retries=args.rate_limit_retries,
                    base_delay=args.rate_limit_delay,
                )
                error = ""
            except Exception as exc:  # noqa: BLE001 - save failure and continue.
                answer = ""
                error = str(exc)

            if error and previous.get("submitted_answer") and not previous.get("error"):
                print("  Rerun failed; keeping the previously saved answer.")
            else:
                saved[task_id] = {
                    "task_id": task_id,
                    "submitted_answer": answer,
                    "error": error,
                    "trace": list(agent.last_trace),
                }
            _save_results(args.output, questions, saved)
            status = "READY" if answer and not error else "FAILED"
            print(f"[{index:02d}/{len(selected):02d}] {status} {task_id}")
            if answer:
                print(f"  Agent answer: {answer}")
            if error:
                print(f"  Error: {error}")
            if args.request_delay and index < len(selected):
                time.sleep(args.request_delay)

    ready = len(_successful_answers(args.output))
    print(f"\nAnswers ready: {ready}/{len(questions)}")
    if ready < len(questions):
        print("Rerun the same command to retry failures without repeating successes.")
    else:
        print("All official answers are saved. Submit only when ready with --submit.")
    return 0


def submit_saved_results(args: argparse.Namespace, api_url: str) -> int:
    questions = fetch_official_questions(api_url)
    payload = build_submission_payload(
        username=os.getenv("HF_USERNAME", ""),
        agent_space=os.getenv("HF_AGENT_SPACE", ""),
        questions=questions,
        saved_answers=_successful_answers(args.output),
    )
    endpoint = f"{api_url}/submit"
    print("Official course submission")
    print(f"Hugging Face username: {payload['username']}")
    print(f"Agent source: {payload['agent_code']}")
    print(f"Answers ready: {len(payload['answers'])}")
    print(f"Submission endpoint: {endpoint}")
    print("Submitting because --submit was explicitly provided...")

    response = requests.post(endpoint, json=payload, timeout=120)
    try:
        response.raise_for_status()
    except requests.RequestException as exc:
        detail = response.text[:500]
        print(f"Submission rejected: {exc}")
        if detail:
            print(f"Server response: {detail}")
        return 1

    result = response.json()
    print("Submission accepted: yes")
    print(f"Score: {result.get('score', 'N/A')}%")
    print(f"Correct official questions: {result.get('correct_count', '?')}")
    print(f"Valid answers submitted: {result.get('total_attempted', '?')}")
    if result.get("message"):
        print(f"Message: {result['message']}")
    return 0


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--all",
        action="store_true",
        help="Run every official course question",
    )
    selection.add_argument(
        "--task-id",
        default="",
        help="Run or rerun one official task ID",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of official questions to run for testing (default: 5)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Private resumable results file (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun selected successful answers instead of using the cache",
    )
    parser.add_argument(
        "--rate-limit-retries",
        type=int,
        default=2,
        help="Retries per question after a provider rate limit",
    )
    parser.add_argument(
        "--rate-limit-delay",
        type=float,
        default=20.0,
        help="Initial rate-limit retry delay in seconds",
    )
    parser.add_argument(
        "--request-delay",
        type=float,
        default=0.0,
        help="Optional pause between questions",
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Submit the saved successful answers to the official scorer",
    )
    args = parser.parse_args()

    if args.limit <= 0:
        parser.error("--limit must be greater than zero; use --all for every question")
    if args.rate_limit_retries < 0:
        parser.error("--rate-limit-retries cannot be negative")
    if args.rate_limit_delay < 0 or args.request_delay < 0:
        parser.error("delays cannot be negative")

    api_url = os.getenv("GAIA_API_URL", DEFAULT_API_URL).rstrip("/")
    try:
        if args.submit:
            return submit_saved_results(args, api_url)
        return run_questions(args, api_url)
    except (AgentConfigurationError, OSError, ValueError, requests.RequestException) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
