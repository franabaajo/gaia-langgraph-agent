"""Practice runner: exercises the agent but never submits a score."""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv

from agent import AgentConfigurationError, GaiaAgent, configuration_report


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="?", help="A question to give the agent")
    parser.add_argument("--file", type=Path, help="Optional local attachment")
    parser.add_argument("--check", action="store_true", help="Only show configuration")
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Show tool calls and observations (never private model reasoning)",
    )
    args = parser.parse_args()

    if args.check or not args.question:
        print("\n".join(configuration_report()))
        if not args.question:
            return

    try:
        agent = GaiaAgent()
        if args.file is None:
            answer = agent(question=args.question, task_id="local-practice")
        else:
            source = args.file.resolve()
            if not source.is_file():
                parser.error(f"Attachment does not exist: {source}")
            with tempfile.TemporaryDirectory(prefix="gaia_attachments_local_") as temp_dir:
                copied_file = Path(temp_dir) / source.name
                shutil.copy2(source, copied_file)
                answer = agent(
                    question=args.question,
                    task_id="local-practice",
                    attachment_path=copied_file,
                )
    except AgentConfigurationError as exc:
        print(f"ERROR - {exc}", file=sys.stderr)
        raise SystemExit(1) from None
    if args.trace:
        print("Execution trace:")
        print(agent.trace_text())
        print("\nFinal answer:")
    print(answer)


if __name__ == "__main__":
    main()
