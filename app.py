"""Gradio runner for the Hugging Face Agents Course final assignment.

Most learners should edit ``agent.py`` and ``tools.py``. This file handles the
course API, attachment downloads, Hugging Face login, and answer submission.
"""

from __future__ import annotations

import os
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Any

import gradio as gr
import pandas as pd
import requests
from dotenv import load_dotenv
from PIL import Image

from agent import (
    AgentConfigurationError,
    GaiaAgent,
    agent_graph_mermaid,
    configuration_report,
    render_agent_graph_png,
)

load_dotenv()

# Keep this URL unless the course changes its scoring service.
DEFAULT_API_URL = os.getenv(
    "GAIA_API_URL", "https://agents-course-unit4-scoring.hf.space"
).rstrip("/")
REQUEST_TIMEOUT = 30
RUNNING_IN_SPACE = bool(os.getenv("SPACE_ID", "").strip())


def download_attachment(item: dict[str, Any], destination: Path) -> Path | None:
    """Download the official attachment for one task, if it has one."""

    file_name = item.get("file_name")
    task_id = item.get("task_id")
    if not file_name or not task_id:
        return None

    # basename prevents a server-supplied filename from escaping destination.
    safe_name = Path(str(file_name)).name
    output_path = destination / f"{task_id}_{safe_name}"
    response = requests.get(
        f"{DEFAULT_API_URL}/files/{task_id}",
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    output_path.write_bytes(response.content)
    return output_path


def fetch_questions() -> list[dict[str, Any]]:
    """Fetch the current evaluation questions from the course API."""

    response = requests.get(f"{DEFAULT_API_URL}/questions", timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    questions = response.json()
    if not isinstance(questions, list) or not questions:
        raise RuntimeError("The questions endpoint returned no questions.")
    return questions


def make_agent() -> GaiaAgent:
    """Create the agent once per evaluation run."""

    return GaiaAgent()


def _profile_username(profile: gr.OAuthProfile | None) -> str:
    """Read a Hugging Face username from Gradio's OAuth profile."""

    if profile is None:
        return ""
    username = getattr(profile, "username", "")
    if not username and isinstance(profile, dict):
        username = profile.get("username", "")
    return str(username).strip()


def _space_owner_username() -> str:
    """Return the user allowed to spend this Space's model quota."""

    configured_owner = os.getenv("SPACE_OWNER_USERNAME", "").strip()
    if configured_owner:
        return configured_owner
    space_id = os.getenv("SPACE_ID", "").strip()
    if "/" in space_id:
        return space_id.split("/", 1)[0].strip()
    return ""


def _space_model_access_error(profile: gr.OAuthProfile | None) -> str | None:
    """Protect paid or quota-limited model calls in a public Space."""

    if not os.getenv("SPACE_ID"):
        return None
    username = _profile_username(profile)
    if not username:
        return "Please log in to Hugging Face first."
    owner = _space_owner_username()
    if not owner:
        return (
            "The Space owner could not be determined. Set SPACE_OWNER_USERNAME "
            "as a repository variable."
        )
    if username.casefold() != owner.casefold():
        return (
            "Only the Space owner can launch model calls in this deployment. "
            "Duplicate the public Space and add your own API key to use it."
        )
    return None


def run_and_submit_all(profile: gr.OAuthProfile | None):
    """Run every official question and submit the resulting answer list."""

    access_error = _space_model_access_error(profile)
    if access_error:
        return access_error, None

    space_id = os.getenv("SPACE_ID")
    if not space_id:
        return (
            (
                "SPACE_ID is missing. Deploy this repository as a Hugging Face Space "
                "before submitting."
            ),
            None,
        )

    try:
        agent = make_agent()
        questions = fetch_questions()
    except AgentConfigurationError as exc:
        return f"Configuration error: {exc}", None
    except Exception as exc:  # noqa: BLE001 - surface startup failures in the UI.
        return f"Could not start evaluation: {exc}", None

    answers_payload: list[dict[str, str]] = []
    results_log: list[dict[str, str]] = []

    # Attachments live only for this run and are deleted automatically afterward.
    with tempfile.TemporaryDirectory(prefix="gaia_attachments_") as temp_dir:
        attachment_dir = Path(temp_dir)

        for number, item in enumerate(questions, start=1):
            task_id = str(item.get("task_id", ""))
            question = item.get("question")
            if not task_id or not isinstance(question, str):
                continue

            try:
                attachment_path = download_attachment(item, attachment_dir)
                answer = agent(
                    question=question,
                    task_id=task_id,
                    attachment_path=attachment_path,
                )
                answers_payload.append(
                    {"task_id": task_id, "submitted_answer": answer}
                )
                results_log.append(
                    {
                        "#": str(number),
                        "Task ID": task_id,
                        "Attachment": item.get("file_name") or "",
                        "Question": question,
                        "Submitted Answer": answer,
                        "Status": "ready",
                    }
                )
            except Exception as exc:  # noqa: BLE001 - continue the evaluation batch.
                # Keep processing the remaining questions. An empty answer is still
                # logged so you can see exactly which capability needs work.
                answers_payload.append(
                    {"task_id": task_id, "submitted_answer": ""}
                )
                results_log.append(
                    {
                        "#": str(number),
                        "Task ID": task_id,
                        "Attachment": item.get("file_name") or "",
                        "Question": question,
                        "Submitted Answer": "",
                        "Status": f"ERROR: {exc}",
                    }
                )

    if not answers_payload:
        return "The agent produced no answers.", pd.DataFrame(results_log)

    submission = {
        "username": profile.username.strip(),
        "agent_code": f"https://huggingface.co/spaces/{space_id}/tree/main",
        "answers": answers_payload,
    }

    try:
        response = requests.post(
            f"{DEFAULT_API_URL}/submit",
            json=submission,
            timeout=120,
        )
        response.raise_for_status()
        result = response.json()
        status = (
            "Submission successful\n"
            f"User: {result.get('username')}\n"
            f"Score: {result.get('score', 'N/A')}% "
            f"({result.get('correct_count', '?')}/"
            f"{result.get('total_attempted', '?')} attempted)\n"
            f"Message: {result.get('message', '')}"
        )
    except requests.RequestException as exc:
        detail = getattr(getattr(exc, "response", None), "text", "")
        status = f"Submission failed: {exc}\n{detail[:500]}"

    return status, pd.DataFrame(results_log)


def test_one_question(
    question: str,
    attachment: str | None,
    profile: gr.OAuthProfile | None = None,
):
    """Run one practice question; restrict hosted quota to the Space owner."""

    access_error = _space_model_access_error(profile)
    if access_error:
        return (
            access_error,
            (
                "Model calls are owner-only in the deployed Space. Local practice "
                "remains unrestricted."
            ),
        )
    if not question.strip():
        return "Enter a question first.", "No run yet."
    agent = None
    try:
        agent = make_agent()
        path = Path(attachment) if attachment else None
        answer = agent(question=question, task_id="practice", attachment_path=path)
        return answer, agent.trace_text()
    except Exception as exc:  # noqa: BLE001 - present tool/provider errors to the user.
        trace = agent.trace_text() if agent is not None else "The agent did not start."
        return f"ERROR: {exc}", trace


def show_configuration() -> str:
    """Return a learner-friendly environment checklist."""

    return "\n".join(configuration_report())


def show_agent_graph():
    """Render the real LangGraph topology and always return its Mermaid source."""

    source = agent_graph_mermaid()
    try:
        image = Image.open(BytesIO(render_agent_graph_png()))
        image.load()
        return image, source, "Graph rendered from the workflow in agent.py."
    except Exception as exc:  # noqa: BLE001 - Mermaid rendering is optional.
        return (
            None,
            source,
            (
                "PNG rendering was unavailable, but the Mermaid source below is "
                f"complete. Details: {exc}"
            ),
        )


with gr.Blocks(title="Guided LangGraph GAIA Agent") as demo:
    gr.Markdown(
        """
        # Guided LangGraph GAIA Agent

        Start with **Configuration check**, then use **Practice** while editing
        `agent.py` and `tools.py`. The active model backend and graph profile
        come from `.env` (Gemini and Ollama default to the `lean` profile).
        Only use **Evaluate & submit** when the practice
        answers contain the answer alone—no explanation or `FINAL ANSWER:` label.
        """
    )

    with gr.Tab("1. Configuration"):
        config_button = gr.Button("Check configuration")
        config_output = gr.Textbox(label="Checklist", lines=14, interactive=False)
        config_button.click(show_configuration, outputs=config_output)

    with gr.Tab("2. Practice (no submission)"):
        gr.Markdown(
            "Local practice needs no login. In the deployed public Space, only "
            "the Space owner's Hugging Face account can launch model calls."
        )
        if RUNNING_IN_SPACE:
            gr.LoginButton()
        practice_question = gr.Textbox(
            label="Question",
            lines=5,
            placeholder="Paste a GAIA-style question here...",
        )
        practice_file = gr.File(label="Optional attachment", type="filepath")
        practice_button = gr.Button("Run agent")
        practice_output = gr.Textbox(label="Agent answer", lines=4)
        practice_trace = gr.Textbox(
            label="Execution trace (tool calls and results, not private reasoning)",
            lines=12,
            interactive=False,
        )
        practice_button.click(
            test_one_question,
            inputs=[practice_question, practice_file],
            outputs=[practice_output, practice_trace],
        )

    with gr.Tab("3. View LangGraph"):
        gr.Markdown(
            "This diagram is generated from the workflow in `agent.py`, so it "
            "updates when you add nodes or change edges. PNG rendering uses the "
            "Mermaid rendering service; the source remains available if it is offline."
        )
        graph_button = gr.Button("Render current graph")
        graph_status = gr.Textbox(label="Status", interactive=False)
        graph_image = gr.Image(label="LangGraph workflow", interactive=False)
        graph_source = gr.Code(label="Mermaid source", language=None, interactive=False)
        graph_button.click(
            show_agent_graph,
            outputs=[graph_image, graph_source, graph_status],
        )

    with gr.Tab("4. Evaluate & submit"):
        gr.Markdown(
            "This runs all course questions and updates the leaderboard only if "
            "the score improves. The Space must be public, but model calls are "
            "restricted to its owner."
        )
        if RUNNING_IN_SPACE:
            gr.LoginButton()
        run_button = gr.Button("Run evaluation and submit all answers", variant="primary")
        status_output = gr.Textbox(label="Run status", lines=6, interactive=False)
        results_table = gr.DataFrame(label="Questions and answers", wrap=True)
        run_button.click(
            run_and_submit_all,
            outputs=[status_output, results_table],
        )


if __name__ == "__main__":
    demo.launch(debug=True, share=False)
