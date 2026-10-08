"""Export the LangGraph workflow as Mermaid source and, when possible, a PNG."""

from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from agent import agent_graph_mermaid, render_agent_graph_png


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("agent_graph.mmd"),
        help="Mermaid output path (default: agent_graph.mmd)",
    )
    parser.add_argument(
        "--png",
        type=Path,
        default=Path("agent_graph.png"),
        help="PNG output path (default: agent_graph.png)",
    )
    parser.add_argument(
        "--profile",
        choices=("lean", "robust"),
        help="Graph profile to export (defaults to the active .env selection)",
    )
    parser.add_argument(
        "--source-only",
        action="store_true",
        help="Write Mermaid source without contacting the PNG rendering service",
    )
    args = parser.parse_args()

    mermaid = agent_graph_mermaid(args.profile)
    args.source.write_text(mermaid, encoding="utf-8")
    print(f"Wrote Mermaid graph: {args.source.resolve()}")

    if args.source_only:
        return

    try:
        args.png.write_bytes(render_agent_graph_png(args.profile))
        print(f"Wrote PNG graph: {args.png.resolve()}")
    except Exception as exc:  # noqa: BLE001 - source export still succeeds offline.
        print(f"Could not render the PNG: {exc}")
        print(
            "The .mmd file is still valid. Preview it with a Mermaid extension "
            "or paste it into https://mermaid.live/."
        )


if __name__ == "__main__":
    main()
