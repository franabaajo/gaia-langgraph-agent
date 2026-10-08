---
title: LangGraph GAIA Agent
emoji: 🧭
colorFrom: indigo
colorTo: blue
sdk: static
app_file: index.html
pinned: false
---

# LangGraph GAIA Agent

A tool-using question-answering agent built with LangGraph for the Hugging Face Agents Course Unit 4 final assignment. It handles web research, calculations, and multimodal attachments through bounded workflows and task-specific tools.

**Official result:** 35% (7/20) on the course scorer, meeting the course completion threshold. The certificate has been claimed.

## How it works

- Routes questions to research, calculation, attachment, or general workflows.
- Uses web search with fallback providers, prioritizes primary sources, and extracts PDFs.
- Supports document, spreadsheet, code, audio, and video tasks; Gemini handles visual YouTube analysis.
- Uses bounded tool loops and answer cleanup for exact-match scoring.
- Includes regression tests for routing, tool behavior, answer handling, and scorer submission.

LangGraph profiles control the workflow. `lean` reduces model calls and latency; `robust` adds planning and verification. Model backends include Gemini, Hugging Face Inference Providers, and Ollama. Ollama is experimental: the tested 8B model did not perform reliably on GAIA.

## Run locally

Python 3.12 is recommended. Install dependencies and configure a backend in `.env` (see `.env.example`).

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
python run_local.py --check
```

For Gemini, add `GEMINI_API_KEY` to `.env`. Run a sample question with:

```powershell
python run_local.py --trace "What is the capital of France?"
```

Run the tests with:

```powershell
python -m unittest
```

## Course evaluation

`submit_results.py` runs the official course questions locally and saves successful answers in a resumable, ignored JSON file. It never sends reference answers to the model. Evaluation and submission are separate commands; `--submit` explicitly sends saved answers to the official scorer.

If the course attachment endpoint returns 404, the runner downloads only the requested attachment from the gated GAIA validation directory. Accept the GAIA dataset terms and authenticate with `hf auth login`; downloaded files remain in the private Hugging Face cache.

```powershell
python submit_results.py --limit 5
python submit_results.py --all
python submit_results.py --submit
```

The public [Hugging Face Space](https://huggingface.co/spaces/fran-abaajo/gaia-langgraph-agent) contains the project source. Secrets and local evaluation results are excluded from version control.
