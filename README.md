---
title: Guided LangGraph GAIA Agent
emoji: 🧭
colorFrom: indigo
colorTo: blue
sdk: static
app_file: index.html
pinned: false
---

# Guided LangGraph GAIA Agent

A LangGraph agent for the Hugging Face Agents Course final assignment. It
answers GAIA-style research, calculation, file, audio, and video questions while
keeping tool use and model calls bounded. Benchmark answers are never embedded
in the code or added to model prompts.

## Highlights

- Deterministic routing for research, calculation, attachment, mixed, and
  general questions
- Web search with DDGS fallbacks, primary-source selection, PDF extraction, and
  benchmark-leak filtering
- Audited arithmetic with entity counts, unit conversion, package sizing, and
  rounding validation
- Text, JSON, CSV, Excel, PDF, DOCX, ZIP, Python, audio, and YouTube tools
- Gemini-native visual analysis when a YouTube answer depends on visible frames
- Conservative final-answer extraction for exact-match scoring
- Execution traces, bounded retries, and regression tests
- Switchable Gemini, Hugging Face, and Ollama backends

## Backends and profiles

Backends select the model provider; profiles select the LangGraph workflow.

| Option | Intended use | Main trade-off |
|---|---|---|
| `gemini` | Recommended local-agent backend through Google's hosted API | Requires an API key and has quota limits |
| `huggingface` | Hosted open models through Inference Providers | Availability and cost depend on the provider |
| `ollama` | Experimental local compatibility | The tested 8B model did not perform reliably on GAIA |
| `lean` | Default for Gemini and Ollama | Fewer calls and no separate model verifier |
| `robust` | Default for Hugging Face | Adds planning and bounded verification at higher cost |

Ollama is fully integrated and can exercise the same graph and tools locally.
However, the tested `qwen3:8b` model failed the selected GAIA proxy task and
could return an empty answer with reasoning enabled. Ollama is therefore kept as
an experimental option for stronger tool-capable models on more powerful
hardware, not as a demonstrated GAIA submission backend.

## Setup

Python 3.12 is the safest dependency target, although the project has also run
under Python 3.14 with a Pydantic compatibility warning.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` and configure one backend. Gemini is the recommended starting
point:

```env
MODEL_BACKEND=gemini
AGENT_PROFILE=lean
GEMINI_API_KEY=your_real_key_here
GEMINI_MODEL=gemini-3.8-flash
GEMINI_FALLBACK_MODEL=gemini-3.5-flash
GEMINI_THINKING_LEVEL=low
GEMINI_MAX_OUTPUT_TOKENS=1200
```

Keep secrets only in the local `.env`. The public Static Space runs no Python
and needs no API keys. Never commit `.env`.

Alternative hosted configuration:

```env
MODEL_BACKEND=huggingface
AGENT_PROFILE=robust
HF_TOKEN=your_read_token
GAIA_MODEL=openai/gpt-oss-20b
HF_INFERENCE_PROVIDER=cheapest
```

Optional experimental Ollama configuration:

```env
MODEL_BACKEND=ollama
AGENT_PROFILE=lean
OLLAMA_MODEL=qwen3:8b
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_REASONING=false
OLLAMA_CONTEXT_LENGTH=8192
OLLAMA_NUM_PREDICT=600
```

## Run and test

Check configuration:

```powershell
python run_local.py --check
```

Run one question with an execution trace:

```powershell
python run_local.py --trace "What is the capital of France?"
```

Start the Gradio application:

```powershell
python app.py
```

Run the regression suite:

```powershell
python -m unittest
```

## Free Hugging Face Course Submission

The agent and Gemini inference run locally. The free public Static Space holds
only the source code used by the course scorer for verification.

1. Test five official questions. Successful answers are saved immediately and
   reused on later runs:

   ```powershell
   python submit_results.py --limit 5
   ```

2. Run all 20 official questions. This fetches questions and attachments from
   the course API but never receives or sends reference answers to the model:

   ```powershell
   python submit_results.py --all
   ```

   Rerun the same command after a failure; completed answers are not repeated.
   Use `--task-id TASK_ID --force` only when intentionally replacing one answer.

3. Create a Hugging Face Space with these settings: **Public** visibility and
   **Static** SDK. The CLI equivalent is:

   ```powershell
   hf auth login
   hf repos create YOUR_USERNAME/gaia-langgraph-agent --repo-type space --space-sdk static --public
   ```

4. Commit and push the source. If `origin` still points to the course template,
   preserve it as `template` and add the new Static Space as `origin`:

   ```powershell
   git add --all
   git status --short
   git commit -m "Implement LangGraph GAIA agent"
   git remote rename origin template
   git remote add origin https://huggingface.co/spaces/YOUR_USERNAME/gaia-langgraph-agent
   git push -u origin main
   ```

   `.env`, JSON results, virtual environments, caches, generated graphs, and
   local project notes are excluded by `.gitignore`.

5. Add the following non-secret identifiers to the local `.env`:

   ```env
   HF_USERNAME=YOUR_USERNAME
   HF_AGENT_SPACE=YOUR_USERNAME/gaia-langgraph-agent
   ```

6. Submit the complete saved answer set. Submission occurs only when this
   explicit flag is present:

   ```powershell
   python submit_results.py --submit
   ```

   The command prints the username, public source URL, answer count, endpoint,
   score, correct count, and server message without printing API keys.

7. Confirm the result on the [student leaderboard](https://huggingface.co/spaces/agents-course/Students_leaderboard).
   A score above 30% is eligible for the official certificate. Sign in at the
   [certificate page](https://huggingface.co/spaces/agents-course/Unit4-Final-Certificate),
   enter the name to print, and select **Get My Certificate**.

## Local proxy evaluation

The optional local evaluator requires a separate GAIA `metadata.jsonl` file,
which is not included in this repository. Reference answers are used only for
scoring after inference and are never added to the model prompt. The evaluator
does not submit anything to the official leaderboard.

```powershell
python evaluate_local.py --metadata path\to\metadata.jsonl --limit 10
```

Rerun one task without spending quota on the full sample:

```powershell
python evaluate_local.py --metadata path\to\metadata.jsonl --task-id TASK_ID --output single_task_result.json
```

Rate-limit handling can be adjusted with `--rate-limit-retries`,
`--rate-limit-delay`, and `--request-delay`. Local proxy performance does not
guarantee the official GAIA score.

## LangGraph profiles

The lean workflow is optimized for latency and quota:

```text
question -> deterministic router -> specialist/tool loop -> finalizer
```

The robust workflow adds explicit planning and one bounded verification pass:

```text
question -> planner -> specialist/tool loop -> verifier -> final answer
```

Both profiles use the same typed tools, recursion limits, and answer
normalization. Export the selected graph with:

```powershell
python visualize_graph.py --source-only
```

## Main files

| File | Responsibility |
|---|---|
| `agent.py` | Model configuration, routing, prompts, graphs, retries, and traces |
| `tools.py` | Research, calculation, attachment, audio, and video tools |
| `answer_normalize.py` | Conservative exact-answer extraction |
| `app.py` | Optional local Gradio practice and graph interface |
| `run_local.py` | Single-question runner and configuration check |
| `evaluate_local.py` | Repeatable unofficial proxy evaluation |
| `submit_results.py` | Resumable official question runner and explicit submission CLI |
| `index.html` | Minimal landing page for the free Static Space |
| `tests/test_starter.py` | Regression coverage |

## Known limitations

- Search providers and publisher pages can be unavailable or block automated
  access.
- Scanned PDFs and image attachments need additional OCR or vision support.
- Gemini visual video analysis and audio transcription depend on external APIs.
- Small local models may support tool calling without having enough reasoning
  capacity for GAIA.
- Exact local results do not guarantee performance on the official hidden set.

The scorer expects only the requested answer, without an explanation or an
`FINAL ANSWER:` prefix.
