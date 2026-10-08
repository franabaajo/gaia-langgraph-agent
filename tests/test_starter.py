from __future__ import annotations

import os
import sys
import tempfile
import types
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableLambda
from pypdf import PdfWriter

from agent import (
    GaiaAgent,
    _classify_task,
    _deterministic_research_followup,
    _deterministic_verification_issue,
    _env_bool,
    _env_positive_int,
    _explicit_literal_answer,
    _gemini_key_is_configured,
    _gemini_sampling_kwargs,
    _gemini_thinking_level,
    _is_transient_gemini_error,
    _latest_answer_candidate,
    _quoted_answer_from_evidence,
    _routed_model_id,
    _selected_agent_profile,
    _selected_backend,
    agent_graph_mermaid,
    build_agent_graph,
    build_execution_trace,
    build_lean_agent_graph,
    configuration_report,
    format_execution_trace,
)
from answer_normalize import normalize_answer, normalize_answer_for_question
from evaluate_local import (
    _answer_with_rate_limit_retries,
    _select_questions,
    answers_match,
)
from submit_results import (
    _save_results,
    _successful_answers,
    build_submission_payload,
    download_attachment,
    run_questions,
)
from tools import (
    ALL_TOOLS,
    _looks_like_pdf_url,
    _source_priority,
    _validate_public_url,
    _youtube_video_id,
    analyze_youtube_video,
    calculate,
    calculate_audited_quantity,
    fetch_pdf,
    fetch_webpage,
    research_primary_source,
    reset_research_session,
    run_python_attachment,
    search_web,
    set_task_question,
)


class AnswerNormalizationTests(unittest.TestCase):
    def test_removes_answer_prefix(self):
        self.assertEqual(normalize_answer(" FINAL ANSWER: Paris "), "Paris")

    def test_preserves_meaningful_formatting(self):
        self.assertEqual(normalize_answer("$123.00"), "$123.00")

    def test_removes_redundant_unit_when_question_supplies_output_unit(self):
        question = (
            "What was the volume in m^3 of the fish bag calculated in the paper?"
        )
        self.assertEqual(
            normalize_answer_for_question(question, "0.1777 m^3"), "0.1777"
        )

    def test_matches_superscript_and_caret_unit_notation(self):
        question = "What was the volume in m³ of the container?"
        self.assertEqual(
            normalize_answer_for_question(question, "0.1777 m^3"), "0.1777"
        )

    def test_preserves_unit_when_question_explicitly_requests_it(self):
        question = "What is the volume? Include the unit m^3 in your answer."
        self.assertEqual(
            normalize_answer_for_question(question, "0.1777 m^3"),
            "0.1777 m^3",
        )

    def test_preserves_unit_mentioned_only_as_input_context(self):
        question = "A parcel has a mass of 5 kg. What is the doubled mass?"
        self.assertEqual(
            normalize_answer_for_question(question, "10 kg"), "10 kg"
        )

    def test_preserves_required_time_qualifier(self):
        question = (
            "Express the arrival in 12-hour format and include whether it is AM "
            "or PM."
        )
        self.assertEqual(
            normalize_answer_for_question(question, "6:41 PM"), "6:41 PM"
        )

    def test_removes_decorative_markdown_from_short_answer(self):
        self.assertEqual(normalize_answer("*Example*"), "Example")

    def test_extracts_final_standalone_number_from_explanation(self):
        response = "The relevant player had 71 walks and 519 at bats.\n\n519"
        self.assertEqual(
            normalize_answer_for_question("How many at bats did he have?", response),
            "519",
        )

    def test_extracts_final_numeric_output_from_explanation(self):
        response = "The loop updates the value several times.\n\n20"
        self.assertEqual(
            normalize_answer_for_question(
                "What is the final numeric output from the attached Python code?",
                response,
            ),
            "20",
        )

    def test_extracts_conclusion_count_without_using_earlier_dates(self):
        response = (
            "The albums released between 2000 and 2009 were A, B, and C. "
            "Therefore, 3 studio albums were published."
        )
        self.assertEqual(
            normalize_answer_for_question(
                "How many studio albums were published between 2000 and 2009?",
                response,
            ),
            "3",
        )

    def test_removes_source_user_label_from_who_answer(self):
        self.assertEqual(
            normalize_answer_for_question(
                "Who nominated the article?", "User: FunkMonk"
            ),
            "FunkMonk",
        )

    def test_extracts_numeric_tail_with_unicode_unit(self):
        response = "The calculation is complete.\n\n25 µg"
        self.assertEqual(
            normalize_answer_for_question("How much material was measured?", response),
            "25 µg",
        )

    def test_extracts_labeled_answer_after_evidence(self):
        self.assertEqual(
            normalize_answer("Evidence from the source.\nFINAL ANSWER: Paris"),
            "Paris",
        )

    def test_extracts_leading_labeled_answer_before_evidence(self):
        self.assertEqual(
            normalize_answer("Answer: 3\nEvidence: visible at 00:42."),
            "3",
        )

    def test_expands_city_abbreviation_only_when_requested(self):
        question = "Give the city name without abbreviations."
        self.assertEqual(
            normalize_answer_for_question(question, "St. Louis"),
            "Saint Louis",
        )
        self.assertEqual(
            normalize_answer_for_question("Give the city name.", "St. Louis"),
            "St. Louis",
        )


class LocalEvaluationTests(unittest.TestCase):
    def test_exact_match_ignores_case_and_outer_whitespace(self):
        self.assertTrue(answers_match("  Paris ", "paris"))

    def test_exact_match_rejects_extra_explanation(self):
        self.assertFalse(answers_match("Paris, because it is the capital", "Paris"))

    def test_rate_limit_retries_one_question_with_backoff(self):
        agent = MagicMock(
            side_effect=[RuntimeError("429 RESOURCE_EXHAUSTED quota"), "Paris"]
        )
        with (
            patch("evaluate_local.time.sleep") as sleep,
            patch("builtins.print"),
        ):
            answer = _answer_with_rate_limit_retries(
                agent,
                question="What is the capital of France?",
                task_id="test",
                retries=2,
                base_delay=3,
            )
        self.assertEqual(answer, "Paris")
        sleep.assert_called_once_with(3)
        self.assertEqual(agent.call_count, 2)

    def test_task_id_selects_only_the_requested_question(self):
        questions = [
            {"task_id": "first", "Question": "One"},
            {"task_id": "target", "Question": "Two"},
            {"task_id": "third", "Question": "Three"},
        ]
        selected = _select_questions(
            questions,
            task_id="target",
            seed=42,
            limit=10,
        )
        self.assertEqual([item["task_id"] for item in selected], ["target"])


class CourseSubmissionTests(unittest.TestCase):
    def test_attachment_404_falls_back_to_private_gaia_dataset(self):
        response = MagicMock(status_code=404)
        with tempfile.TemporaryDirectory() as temp_dir:
            cached_file = Path(temp_dir) / "task.png"
            cached_file.write_bytes(b"image")
            with (
                patch("submit_results.requests.get", return_value=response),
                patch(
                    "submit_results.hf_hub_download",
                    return_value=str(cached_file),
                ) as hub_download,
                patch("builtins.print"),
            ):
                result = download_attachment(
                    {"task_id": "task-id", "file_name": "task.png"},
                    Path(temp_dir),
                    "https://course.example",
                )

        self.assertEqual(result, cached_file)
        hub_download.assert_called_once_with(
            repo_id="gaia-benchmark/GAIA",
            filename="2023/validation/task.png",
            repo_type="dataset",
        )

    def test_attachment_filename_cannot_escape_validation_directory(self):
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            self.assertRaisesRegex(ValueError, "Unsafe official attachment"),
        ):
            download_attachment(
                {"task_id": "task-id", "file_name": "../answer.json"},
                Path(temp_dir),
                "https://course.example",
            )

    def test_submission_payload_matches_official_course_schema(self):
        questions = [{"task_id": "task-a"}, {"task_id": "task-b"}]
        payload = build_submission_payload(
            username="student",
            agent_space="student/public-agent",
            questions=questions,
            saved_answers={"task-a": "alpha", "task-b": "beta"},
        )

        self.assertEqual(
            payload,
            {
                "username": "student",
                "agent_code": (
                    "https://huggingface.co/spaces/student/public-agent/tree/main"
                ),
                "answers": [
                    {"task_id": "task-a", "submitted_answer": "alpha"},
                    {"task_id": "task-b", "submitted_answer": "beta"},
                ],
            },
        )

    def test_submission_payload_allows_successful_partial_answers(self):
        payload = build_submission_payload(
            username="student",
            agent_space="student/public-agent",
            questions=[{"task_id": "task-a"}, {"task_id": "task-b"}],
            saved_answers={"task-a": "alpha"},
        )
        self.assertEqual(
            payload["answers"],
            [{"task_id": "task-a", "submitted_answer": "alpha"}],
        )

    def test_submission_payload_requires_at_least_one_answer(self):
        with self.assertRaisesRegex(ValueError, "No successful official answers"):
            build_submission_payload(
                username="student",
                agent_space="student/public-agent",
                questions=[{"task_id": "task-a"}],
                saved_answers={},
            )

    def test_submission_payload_normalizes_cached_answer(self):
        payload = build_submission_payload(
            username="student",
            agent_space="student/public-agent",
            questions=[
                {"task_id": "task-a", "question": "How many albums?"}
            ],
            saved_answers={"task-a": "Therefore, 3 albums were published."},
        )
        self.assertEqual(payload["answers"][0]["submitted_answer"], "3")

    def test_official_results_resume_only_successful_answers(self):
        questions = [{"task_id": "task-a"}, {"task_id": "task-b"}]
        saved = {
            "task-a": {
                "task_id": "task-a",
                "submitted_answer": "alpha",
                "error": "",
                "trace": [],
            },
            "task-b": {
                "task_id": "task-b",
                "submitted_answer": "",
                "error": "temporary failure",
                "trace": [],
            },
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "results.json"
            _save_results(output, questions, saved)
            ready = _successful_answers(output)
            raw_output = output.read_text(encoding="utf-8")

        self.assertEqual(ready, {"task-a": "alpha"})
        self.assertNotIn("Final answer", raw_output)
        self.assertNotIn('"question":', raw_output.lower())
        self.assertNotIn("reference", raw_output.lower())

    def test_official_local_run_resumes_without_submitting_or_repeating(self):
        questions = [
            {"task_id": "task-a", "question": "Return only alpha.", "file_name": ""}
        ]
        fake_agent = MagicMock(return_value="alpha")
        fake_agent.last_trace = []
        with tempfile.TemporaryDirectory() as temp_dir:
            args = types.SimpleNamespace(
                task_id="",
                all=False,
                limit=1,
                output=Path(temp_dir) / "official.json",
                force=False,
                rate_limit_retries=0,
                rate_limit_delay=0.0,
                request_delay=0.0,
            )
            with (
                patch("submit_results.fetch_official_questions", return_value=questions),
                patch("submit_results.GaiaAgent", return_value=fake_agent),
                patch(
                    "submit_results.configuration_report",
                    return_value=[
                        "OK - MODEL_BACKEND=gemini",
                        "OK - AGENT_PROFILE=lean",
                    ],
                ),
                patch("submit_results.requests.post") as post,
                patch("builtins.print"),
            ):
                self.assertEqual(run_questions(args, "https://course.example"), 0)
                self.assertEqual(run_questions(args, "https://course.example"), 0)

        fake_agent.assert_called_once()
        post.assert_not_called()


class ToolTests(unittest.TestCase):
    def test_python_attachment_does_not_inherit_api_secrets(self):
        completed = types.SimpleNamespace(returncode=0, stdout="ok", stderr="")
        with (
            patch.dict(
                os.environ,
                {
                    "GEMINI_API_KEY": "gemini-secret",
                    "HF_TOKEN": "hf-secret",
                    "OPENAI_API_KEY": "openai-secret",
                    "SAFE_SETTING": "allowed",
                },
            ),
            patch("tools._official_course_file", return_value=Path(__file__)),
            patch("tools.subprocess.run", return_value=completed) as run,
        ):
            result = run_python_attachment.invoke({"file_path": __file__})

        child_environment = run.call_args.kwargs["env"]
        self.assertEqual(result.splitlines()[0], "Exit code: 0")
        self.assertNotIn("GEMINI_API_KEY", child_environment)
        self.assertNotIn("HF_TOKEN", child_environment)
        self.assertNotIn("OPENAI_API_KEY", child_environment)
        self.assertEqual(child_environment["SAFE_SETTING"], "allowed")

    def test_calculator(self):
        self.assertEqual(
            calculate.invoke({"expression": "ceil(1002 * 0.04)"}),
            "41",
        )

    def test_audited_quantity_derives_totals_from_role_lines(self):
        result = calculate_audited_quantity.invoke(
            {
                "groups": [
                    {
                        "label": "technicians",
                        "included_items": [
                            {"description": "morning shift", "count": 2},
                            {"description": "evening shift", "count": 3},
                        ],
                        "amount_per_member": 4,
                    },
                    {
                        "label": "supervisors",
                        "included_items": [
                            {"description": "floor supervisors", "count": 2},
                        ],
                        "amount_per_member": 1,
                    },
                ],
                "excluded_items": [
                    {"description": "absent technician", "count": 1},
                ],
                "converted_units_per_base_unit": 2,
                "package_size_in_converted_units": 10,
                "round_up": True,
            }
        )
        self.assertIn("technicians:", result)
        self.assertIn("-> 5 x 4", result)
        self.assertIn("Final result: 5", result)

    def test_audited_quantity_rejects_included_excluded_overlap(self):
        with self.assertRaisesRegex(ValueError, "both included and excluded"):
            calculate_audited_quantity.invoke(
                {
                    "groups": [
                        {
                            "label": "participants",
                            "included_items": [
                                {"description": "night shift", "count": 4}
                            ],
                            "amount_per_member": 1.5,
                        }
                    ],
                    "excluded_items": [
                        {"description": "night shift", "count": 1}
                    ],
                }
            )

    def test_audited_quantity_rejects_reversed_stated_conversion(self):
        set_task_question(
            "Each person uses one widget. The average widget weighs half a "
            "kilogram, and widgets are sold in 5-kilogram boxes."
        )
        try:
            with self.assertRaisesRegex(
                ValueError, "converted_units_per_base_unit cannot be 1"
            ):
                calculate_audited_quantity.invoke(
                    {
                        "groups": [
                            {
                                "label": "people",
                                "included_items": [
                                    {"description": "attendees", "count": 18}
                                ],
                                "amount_per_member": 1,
                            }
                        ],
                        "excluded_items": [],
                        "converted_units_per_base_unit": 1,
                        "package_size_in_converted_units": 5,
                        "round_up": True,
                    }
                )
        finally:
            set_task_question("")

    def test_youtube_id_from_url(self):
        self.assertEqual(
            _youtube_video_id("https://www.youtube.com/watch?v=L1vXCYZAYYM"),
            "L1vXCYZAYYM",
        )

    def test_private_web_targets_are_rejected(self):
        with self.assertRaises(ValueError):
            _validate_public_url("http://127.0.0.1/private")

    def test_pdf_tool_rejects_private_targets(self):
        with self.assertRaises(ValueError):
            fetch_pdf.invoke({"url": "http://127.0.0.1/private.pdf"})

    def test_pdf_tool_is_bound_to_agent(self):
        self.assertIn("fetch_pdf", [item.name for item in ALL_TOOLS])

    def test_pdf_tool_downloads_and_opens_a_pdf(self):
        document = BytesIO()
        writer = PdfWriter()
        writer.add_blank_page(width=72, height=72)
        writer.write(document)
        pdf_bytes = document.getvalue()

        response = MagicMock()
        response.is_redirect = False
        response.is_permanent_redirect = False
        response.status_code = 200
        response.headers = {"Content-Length": str(len(pdf_bytes))}
        response.iter_content.return_value = [pdf_bytes]
        with (
            patch("tools._validate_public_url"),
            patch("tools.requests.get", return_value=response),
        ):
            result = fetch_pdf.invoke({"url": "https://example.com/paper.pdf"})

        self.assertIn("no extractable text", result)
        response.close.assert_called_once()

    def test_search_blocks_duplicate_query_within_question(self):
        reset_research_session()
        result = {
            "title": "Official paper",
            "href": "https://journal.example.org/paper",
            "body": "Abstract",
        }
        with patch("tools.DDGS") as ddgs:
            ddgs.return_value.text.return_value = [result]
            first = search_web.invoke({"query": "exact paper title"})
            second = search_web.invoke({"query": "  Exact  Paper Title "})
        self.assertIn("Official paper", first)
        self.assertIn("DUPLICATE SEARCH BLOCKED", second)

    def test_search_uses_explicit_ddgs_fallback_backend(self):
        reset_research_session()
        result = {
            "title": "Official result",
            "href": "https://example.edu/source",
            "body": "Evidence",
        }
        with patch("tools.DDGS") as ddgs:
            ddgs.return_value.text.side_effect = [RuntimeError("no results"), [result]]
            output = search_web.invoke({"query": "rare factual query"})
        self.assertIn("Official result", output)
        self.assertEqual(ddgs.return_value.text.call_count, 2)
        self.assertEqual(
            ddgs.return_value.text.call_args_list[1].kwargs["backend"],
            "duckduckgo,brave,google",
        )

    def test_search_filters_benchmark_copies(self):
        reset_research_session()
        leaked = {
            "title": "GAIA task instruction",
            "href": "https://github.com/example/datasets/gaia/task/instruction.md",
            "body": "Final answer: leaked",
        }
        source = {
            "title": "Publisher article",
            "href": "https://journal.example.edu/article/1",
            "body": "Abstract",
        }
        with patch("tools.DDGS") as ddgs:
            ddgs.return_value.text.return_value = [leaked, source]
            output = search_web.invoke({"query": "article title"})
        self.assertIn("Publisher article", output)
        self.assertNotIn("instruction.md", output)

    def test_search_falls_back_when_first_backend_only_finds_benchmark_copy(self):
        reset_research_session()
        leaked = {
            "title": "GAIA instruction",
            "href": "https://github.com/example/datasets/gaia/task/instruction.md",
            "body": "task_id and Final answer",
        }
        source = {
            "title": "Independent source",
            "href": "https://example.edu/source",
            "body": "Evidence",
        }
        with patch("tools.DDGS") as ddgs:
            ddgs.return_value.text.side_effect = [[leaked], [source]]
            output = search_web.invoke({"query": "underlying subject"})
        self.assertIn("Independent source", output)
        self.assertEqual(ddgs.return_value.text.call_count, 2)

    def test_search_rejects_task_copy_on_unrelated_page(self):
        question = (
            "In a named research article, what unusual descriptive word was quoted "
            "from two different authors to criticize a recurring depiction?"
        )
        copied = {
            "title": "PDF research article unusual descriptive word quoted authors",
            "href": "https://unrelated.example/files/document.pdf",
            "body": question,
        }
        source = {
            "title": "Original journal article",
            "href": "https://journal.example.edu/article/12",
            "body": "Abstract about literary depictions",
        }
        reset_research_session()
        set_task_question(question)
        try:
            with patch("tools.DDGS") as ddgs:
                ddgs.return_value.text.side_effect = [[copied], [source]]
                output = search_web.invoke({"query": "author article depiction"})
        finally:
            set_task_question("")
        self.assertIn("Original journal article", output)
        self.assertNotIn("unrelated.example", output)

    def test_search_rejects_search_engine_error_pages(self):
        reset_research_session()
        bad = {
            "title": "query words",
            "href": "https://www.google.com/search?q=query+words",
            "body": "[Error 500]",
        }
        source = {
            "title": "Useful source",
            "href": "https://example.edu/useful",
            "body": "Evidence",
        }
        with patch("tools.DDGS") as ddgs:
            ddgs.return_value.text.side_effect = [[bad], [source]]
            output = search_web.invoke({"query": "query words"})
        self.assertIn("Useful source", output)
        self.assertNotIn("Error 500", output)

    def test_search_ranks_specific_article_above_generic_editorial(self):
        reset_research_session()
        poisoned_pdf = {
            "title": "dana rivera may 2018 clockwork birds journal",
            "href": (
                "https://unrelated.example/medical/"
                "vaccine-requirements-applicant-age.pdf"
            ),
            "body": "Dana Rivera wrote about clockwork birds in May 2018.",
        }
        marketplace = {
            "title": "dana rivera may 2018 clockwork birds quoted word",
            "href": "https://www.etsy.com/market/dana_rivera_clockwork_birds",
            "body": "Dana Rivera May 2018 clockwork birds article quoted authors.",
        }
        editorial = {
            "title": "Editorial 2/2014",
            "href": "https://journal.example/articles/editorial-22014/",
            "body": "This issue includes an article about clockwork birds.",
        }
        article = {
            "title": "Clockwork Birds in Modern Children's Literature",
            "href": "https://journal.example/article/view/164228",
            "body": "Dana Rivera, published May 15, 2018.",
        }
        with patch("tools.DDGS") as ddgs:
            ddgs.return_value.text.return_value = [
                poisoned_pdf,
                marketplace,
                editorial,
                article,
            ]
            output = search_web.invoke(
                {"query": "Dana Rivera May 2018 clockwork birds journal"}
            )
        self.assertLess(output.index("Clockwork Birds"), output.index("Editorial"))
        self.assertLess(output.index("Clockwork Birds"), output.index("vaccine"))
        self.assertNotIn("etsy.com", output)

    def test_search_reports_ddgs_exhaustion_without_hidden_model_call(self):
        reset_research_session()
        with patch(
            "tools._ddgs_text_search", side_effect=RuntimeError("no results")
        ):
            output = search_web.invoke({"query": "rare scholarly article"})
        self.assertIn("No search results found after DDGS fallbacks", output)

    def test_download_text_url_is_not_misclassified_as_pdf(self):
        self.assertFalse(
            _looks_like_pdf_url(
                "https://archive.example/download/item/document_djvu.txt"
            )
        )
        self.assertTrue(
            _looks_like_pdf_url("https://journal.example/article/download/12/34")
        )

    def test_search_priority_penalizes_aggregators(self):
        publisher_score, _ = _source_priority(
            "https://journals.example.edu/article/123"
        )
        mirror_score, _ = _source_priority("https://core.ac.uk/download/123.pdf")
        self.assertGreater(publisher_score, mirror_score)

    def test_webpage_detects_bot_challenge(self):
        response = MagicMock()
        response.is_redirect = False
        response.is_permanent_redirect = False
        response.status_code = 200
        response.headers = {"Content-Type": "text/html"}
        response.content = b"Making sure you're not a bot! Anubis"
        response.text = response.content.decode()
        with (
            patch("tools._validate_public_url"),
            patch("tools.requests.get", return_value=response),
        ):
            result = fetch_webpage.invoke({"url": "https://journal.example/article"})
        self.assertIn("BLOCKED PAGE", result)

    def test_webpage_exposes_citation_pdf_metadata(self):
        response = MagicMock()
        response.is_redirect = False
        response.is_permanent_redirect = False
        response.status_code = 200
        response.headers = {"Content-Type": "text/html"}
        response.text = (
            '<html><head><meta name="citation_pdf_url" '
            'content="/article/download/123.pdf"></head><body>Paper</body></html>'
        )
        response.content = response.text.encode()
        with (
            patch("tools._validate_public_url"),
            patch("tools.requests.get", return_value=response),
        ):
            result = fetch_webpage.invoke({"url": "https://journal.example/article/123"})
        self.assertIn("https://journal.example/article/download/123.pdf", result)

    def test_primary_research_follows_publisher_pdf_metadata(self):
        ranked_result = (
            "Source priority: likely publisher\n"
            "Title: Paper title\n"
            "URL: https://journal.example/article/123\n"
            "Snippet: Abstract"
        )
        landing_page = (
            "Paper landing page\n\n"
            "Direct PDF links found in page metadata or links:\n"
            "https://journal.example/article/download/123.pdf"
        )
        with (
            patch("tools._run_search_tool", return_value=ranked_result),
            patch("tools._run_webpage_tool", return_value=landing_page),
            patch("tools._run_pdf_tool", return_value="[Page 1]\nExact fact"),
        ):
            result = research_primary_source.invoke({"query": "Paper title"})
        self.assertIn("Selected direct PDF", result)
        self.assertIn("Exact fact", result)

    def test_primary_research_prefers_direct_publisher_pdf(self):
        results = (
            "Source priority: general web\n"
            "Title: Commentary\n"
            "URL: https://example.com/commentary\n"
            "Snippet: Secondary source\n\n"
            "Source priority: likely publisher\n"
            "Title: Paper PDF\n"
            "URL: https://publisher.example/paper.pdf\n"
            "Snippet: Original"
        )
        with (
            patch("tools._run_search_tool", return_value=results),
            patch("tools._run_webpage_tool") as webpage,
            patch("tools._run_pdf_tool", return_value="[Page 1]\nPrimary fact"),
        ):
            output = research_primary_source.invoke({"query": "paper title"})
        self.assertIn("Selected direct source PDF", output)
        webpage.assert_not_called()

    def test_primary_research_opens_relevant_article_before_editorial(self):
        results = (
            "Source priority: likely publisher\n"
            "Title: Editorial 1/2018\n"
            "URL: https://journal.example/editorial/2018\n"
            "Snippet: This issue mentions clockwork birds.\n\n"
            "Source priority: likely publisher\n"
            "Title: Clockwork Birds in Modern Literature\n"
            "URL: https://journal.example/article/42\n"
            "Snippet: Dana Rivera, May 2018."
        )
        article_text = "Specific article text. " * 40
        with (
            patch("tools._run_search_tool", return_value=results),
            patch("tools._run_webpage_tool", return_value=article_text) as webpage,
        ):
            output = research_primary_source.invoke(
                {"query": "Dana Rivera May 2018 clockwork birds"}
            )
        self.assertEqual(
            webpage.call_args_list[0].args[0],
            "https://journal.example/article/42",
        )
        self.assertIn("Specific article text", output)

    def test_gemini_analyzes_youtube_visual_content(self):
        fake_model = MagicMock()
        fake_model.invoke.return_value = AIMessage(
            content="Answer: 3\nEvidence: three birds are visible at 00:42."
        )
        fake_class = MagicMock(return_value=fake_model)
        with (
            patch.dict(
                os.environ,
                {
                    "GEMINI_API_KEY": "test-only-secret-value",
                    "GEMINI_MODEL": "gemini-test-model",
                },
            ),
            patch("langchain_google_genai.ChatGoogleGenerativeAI", fake_class),
        ):
            output = analyze_youtube_video.invoke(
                {
                    "url_or_id": "https://www.youtube.com/watch?v=L1vXCYZAYYM",
                    "question": "How many bird species are on camera simultaneously?",
                }
            )
        self.assertIn("Answer: 3", output)
        content = fake_model.invoke.call_args.args[0][0].content
        media = next(block for block in content if block["type"] == "media")
        self.assertEqual(
            media["file_uri"],
            "https://www.youtube.com/watch?v=L1vXCYZAYYM",
        )


class ConfigurationTests(unittest.TestCase):
    def test_example_token_is_not_treated_as_configured(self):
        with patch.dict(
            os.environ,
            {
                "MODEL_BACKEND": "huggingface",
                "HF_TOKEN": "hf_your_read_token_here",
            },
        ):
            self.assertTrue(
                any(line.startswith("MISSING - HF_TOKEN") for line in configuration_report())
            )

    def test_provider_is_added_to_model_id(self):
        self.assertEqual(
            _routed_model_id("openai/gpt-oss-20b", "groq"),
            "openai/gpt-oss-20b:groq",
        )

    def test_local_alias_selects_ollama(self):
        with patch.dict(os.environ, {"MODEL_BACKEND": "local"}):
            self.assertEqual(_selected_backend(), "ollama")

    def test_google_alias_selects_gemini(self):
        with patch.dict(os.environ, {"MODEL_BACKEND": "google"}):
            self.assertEqual(_selected_backend(), "gemini")

    def test_profile_defaults_to_lean_for_ollama(self):
        with patch.dict(
            os.environ,
            {"MODEL_BACKEND": "ollama", "AGENT_PROFILE": ""},
        ):
            self.assertEqual(_selected_agent_profile(), "lean")

    def test_profile_defaults_to_robust_for_huggingface(self):
        with patch.dict(
            os.environ,
            {"MODEL_BACKEND": "huggingface", "AGENT_PROFILE": ""},
        ):
            self.assertEqual(_selected_agent_profile(), "robust")

    def test_profile_defaults_to_lean_for_gemini(self):
        with patch.dict(
            os.environ,
            {"MODEL_BACKEND": "gemini", "AGENT_PROFILE": ""},
        ):
            self.assertEqual(_selected_agent_profile(), "lean")

    def test_gemini_configuration_reports_key_without_revealing_it(self):
        fake_key = "test-only-secret-value"
        with patch.dict(
            os.environ,
            {
                "MODEL_BACKEND": "gemini",
                "GEMINI_API_KEY": fake_key,
                "GEMINI_MODEL": "gemini-test-model",
                "GEMINI_THINKING_LEVEL": "low",
            },
        ):
            report_text = "\n".join(configuration_report())
        self.assertIn("OK - GEMINI_API_KEY", report_text)
        self.assertIn("INFO - GEMINI_MODEL=gemini-test-model", report_text)
        self.assertNotIn(fake_key, report_text)

    def test_gemini_placeholder_key_is_not_configured(self):
        self.assertFalse(
            _gemini_key_is_configured("replace_with_your_gemini_key")
        )

    def test_gemini_thinking_level_is_validated(self):
        with patch.dict(os.environ, {"GEMINI_THINKING_LEVEL": "low"}):
            self.assertEqual(_gemini_thinking_level(), "low")

    def test_gemini_lite_model_omits_ignored_sampling_setting(self):
        self.assertEqual(_gemini_sampling_kwargs("gemini-3.5-flash-lite"), {})
        self.assertEqual(
            _gemini_sampling_kwargs("gemini-3.5-flash"),
            {"temperature": 1.0},
        )

    def test_gemini_transient_error_detection_is_narrow(self):
        self.assertTrue(
            _is_transient_gemini_error(
                RuntimeError("503 UNAVAILABLE: model is experiencing high demand")
            )
        )
        self.assertFalse(
            _is_transient_gemini_error(RuntimeError("API key not valid"))
        )

    def test_gemini_backend_builds_the_lean_graph(self):
        fake_model = MagicMock()
        fake_model.bind_tools.return_value = RunnableLambda(
            lambda _: AIMessage(content="test")
        )
        fake_class = MagicMock(return_value=fake_model)
        fake_module = types.SimpleNamespace(ChatGoogleGenerativeAI=fake_class)
        with (
            patch.dict(
                os.environ,
                {
                    "MODEL_BACKEND": "gemini",
                    "AGENT_PROFILE": "lean",
                    "GEMINI_API_KEY": "test-only-secret-value",
                    "GEMINI_MODEL": "gemini-test-model",
                    "GEMINI_FALLBACK_MODEL": "",
                    "GEMINI_THINKING_LEVEL": "low",
                },
            ),
            patch.dict(sys.modules, {"langchain_google_genai": fake_module}),
        ):
            agent = GaiaAgent()
        self.assertEqual(agent.backend, "gemini")
        self.assertEqual(agent.profile, "lean")
        fake_class.assert_called_once()
        self.assertEqual(fake_class.call_args.kwargs["thinking_level"], "low")

    def test_gemini_503_restarts_with_fallback_graph(self):
        def unavailable(_):
            raise RuntimeError("503 UNAVAILABLE: temporary high demand")

        agent = object.__new__(GaiaAgent)
        agent.last_trace = []
        agent.backend = "gemini"
        agent.profile = "lean"
        agent.model_id = "gemini-3.8-flash"
        agent.fallback_model_id = "gemini-3.5-flash"
        agent.graph_step_limit = 12
        agent.graph = RunnableLambda(unavailable)
        agent.fallback_graph = RunnableLambda(
            lambda _: {
                "messages": [AIMessage(content="Paris")],
                "plan": [],
                "task_type": "general",
                "final_answer": "Paris",
                "verification_status": "lean",
            }
        )

        self.assertEqual(agent("What is the capital of France?"), "Paris")
        self.assertIn("MODEL FALLBACK", agent.trace_text())

    def test_agent_applies_question_aware_unit_cleanup(self):
        agent = object.__new__(GaiaAgent)
        agent.last_trace = []
        agent.backend = "gemini"
        agent.profile = "lean"
        agent.model_id = "test-model"
        agent.fallback_model_id = ""
        agent.fallback_graph = None
        agent.graph_step_limit = 16
        agent.graph = RunnableLambda(
            lambda _: {
                "messages": [AIMessage(content="0.1777 m^3")],
                "plan": [],
                "task_type": "mixed",
                "final_answer": "0.1777 m^3",
                "verification_status": "lean",
            }
        )

        question = "What was the volume in m^3 of the fish bag?"
        self.assertEqual(agent(question), "0.1777")
        self.assertIn("0.1777", agent.trace_text())

    def test_ollama_does_not_require_hf_token_for_chat(self):
        with patch.dict(
            os.environ,
            {"MODEL_BACKEND": "ollama", "HF_TOKEN": ""},
        ):
            report = configuration_report()
            self.assertIn("OPTIONAL - HF_TOKEN", "\n".join(report))

    def test_ollama_tuning_values_are_parsed(self):
        with patch.dict(
            os.environ,
            {"OLLAMA_REASONING": "yes", "OLLAMA_NUM_PREDICT": "700"},
        ):
            self.assertTrue(_env_bool("OLLAMA_REASONING", False))
            self.assertEqual(_env_positive_int("OLLAMA_NUM_PREDICT", 600), 700)


class GraphTests(unittest.TestCase):
    def test_graph_visualization_contains_main_nodes(self):
        diagram = agent_graph_mermaid("robust")
        for node in ("planner", "researcher", "calculator", "tools", "verifier"):
            self.assertIn(node, diagram)

    def test_lean_graph_visualization_has_no_verifier(self):
        diagram = agent_graph_mermaid("lean")
        for node in ("router", "assistant", "tools", "forced_answer", "finalize"):
            self.assertIn(node, diagram)
        self.assertNotIn("verifier", diagram)

    def test_planner_routes_the_three_failure_types(self):
        self.assertEqual(
            _classify_task("How many whole bags do I need?"), "calculation"
        )
        self.assertEqual(
            _classify_task("What was the volume in the University paper?"),
            "research",
        )
        self.assertEqual(
            _classify_task("What word was quoted in the journal article?"),
            "research",
        )

    def test_factual_counts_route_to_research_not_arithmetic(self):
        self.assertEqual(
            _classify_task(
                "How many at bats did the player with the most walks in 1977 have?"
            ),
            "research",
        )
        self.assertEqual(
            _classify_task("What country had the fewest athletes in the Olympics?"),
            "research",
        )
        self.assertEqual(
            _classify_task("What is the capital of France?"),
            "research",
        )
        self.assertEqual(
            _classify_task("How many more blocks does model A have than model B?"),
            "calculation",
        )

    def test_final_explicit_literal_instruction_takes_precedence(self):
        question = (
            'If something is unclear, write "Fallback". Do not solve the trivia. '
            'Write only the word "Requested". What is 2 + 2?'
        )
        self.assertEqual(_explicit_literal_answer(question), "Requested")
        self.assertEqual(_classify_task(question), "general")

        worker = MagicMock()
        graph = build_lean_agent_graph(
            {
                "researcher": worker,
                "calculator": worker,
                "generalist": worker,
            },
            worker,
        )
        result = graph.invoke(
            {
                "messages": [],
                "question": question,
                "task_id": "literal-test",
                "attachment_path": "",
            }
        )
        self.assertEqual(result["final_answer"], "Requested")
        worker.invoke.assert_not_called()

    def test_visual_youtube_question_routes_directly_to_video_tool(self):
        question = (
            "In https://www.youtube.com/watch?v=L1vXCYZAYYM, what is the highest "
            "number of bird species on camera simultaneously?"
        )
        followup = _deterministic_research_followup(
            {"task_type": "research", "question": question, "messages": []}
        )
        self.assertIsNotNone(followup)
        self.assertEqual(
            followup.tool_calls[0]["name"],
            "analyze_youtube_video",
        )

    def test_research_verifier_requires_a_full_source(self):
        messages = [
            ToolMessage(
                content="Title and snippet only",
                tool_call_id="call-1",
                name="search_web",
            )
        ]
        issue = _deterministic_verification_issue(
            "What word was quoted in the article?", "research", "quartz", messages
        )
        self.assertIn("No full source", issue or "")

    def test_extracts_quoted_word_from_best_matching_source_context(self):
        question = (
            "In Dana Rivera's article, what word was quoted from two different "
            "reviewers in distaste for the nature of clockwork-bird depictions?"
        )
        source = (
            "Pat Lee and Sam Orr both comment with distaste on the increasingly "
            "cuddly, “velvety” nature of clockwork-bird depictions. "
            + ("Unrelated discussion follows. " * 30)
            + 'A later section uses the word “human” in another context.'
        )
        answer = _quoted_answer_from_evidence(
            question,
            [
                ToolMessage(
                    content=source,
                    tool_call_id="source-call",
                    name="fetch_webpage",
                )
            ],
        )
        self.assertEqual(answer, "velvety")

    def test_calculation_verifier_accepts_audited_result(self):
        tool_call = {
            "name": "calculate_audited_quantity",
            "args": {
                "groups": [
                    {
                        "label": "participants",
                        "included_items": [
                            {"description": "day shift", "count": 2}
                        ],
                        "amount_per_member": 1.5,
                    }
                ],
                "excluded_items": [],
                "converted_units_per_base_unit": 0.5,
                "package_size_in_converted_units": 5,
                "round_up": True,
            },
            "id": "call-ledger",
            "type": "tool_call",
        }
        messages = [
            AIMessage(content="", tool_calls=[tool_call]),
            ToolMessage(
                content="Audited count ledger and Final result: 1",
                tool_call_id="call-ledger",
                name="calculate_audited_quantity",
            ),
            AIMessage(content="1"),
        ]
        issue = _deterministic_verification_issue(
            "Two people need 1.5 units each. How many packages? Just give the number.",
            "calculation",
            "1",
            messages,
        )
        self.assertIsNone(issue)

    def test_calculator_result_can_be_recovered_after_a_tool_loop(self):
        candidate = _latest_answer_candidate(
            [
                ToolMessage(
                    content="Audited count ledger:\nBase total: 18\nFinal result: 2",
                    tool_call_id="call-fallback",
                    name="calculate_audited_quantity",
                )
            ]
        )
        self.assertEqual(candidate, "2")

    def test_serialized_tool_call_is_not_accepted_as_the_final_answer(self):
        candidate = _latest_answer_candidate(
            [
                ToolMessage(
                    content="Audited count ledger:\nBase total: 18\nFinal result: 2",
                    tool_call_id="call-fallback",
                    name="calculate_audited_quantity",
                ),
                AIMessage(
                    content=(
                        '<tool_call>{"name":"calculate_audited_quantity",'
                        '"arguments":{}}</tool_call>'
                    )
                ),
            ]
        )
        self.assertEqual(candidate, "2")

    def test_plain_json_tool_call_is_not_accepted_as_the_final_answer(self):
        candidate = _latest_answer_candidate(
            [
                ToolMessage(
                    content="Error: invalid arguments",
                    tool_call_id="call-failed",
                    name="calculate_audited_quantity",
                ),
                AIMessage(
                    content=(
                        '{"name":"calculate_audited_quantity",'
                        '"arguments":{"groups":[]}}'
                    )
                ),
            ]
        )
        self.assertEqual(candidate, "")

    def test_search_observation_dump_is_not_accepted_as_final_answer(self):
        messages = [
            AIMessage(
                content=(
                    "search_web result:\nSource priority: general web\n"
                    "Title: Query\nURL: https://example.com/search\nSnippet: error"
                )
            )
        ]
        self.assertEqual(_latest_answer_candidate(messages), "")

    def test_graph_completes_an_approved_candidate(self):
        worker = RunnableLambda(lambda _: AIMessage(content="Paris"))
        verifier = RunnableLambda(
            lambda _: AIMessage(
                content="VERDICT: APPROVE\nFEEDBACK: Correct and concise."
            )
        )
        graph = build_agent_graph(
            {
                "researcher": worker,
                "calculator": worker,
                "generalist": worker,
            },
            verifier,
        )
        result = graph.invoke(
            {
                "messages": [],
                "question": "Reply with the city Paris.",
                "task_id": "test",
                "attachment_path": "",
            }
        )
        self.assertEqual(result["final_answer"], "Paris")
        self.assertEqual(result["verification_status"], "approved")

    def test_lean_graph_completes_without_a_verifier_call(self):
        worker = RunnableLambda(lambda _: AIMessage(content="Paris"))
        graph = build_lean_agent_graph(
            {
                "researcher": worker,
                "calculator": worker,
                "generalist": worker,
            },
            worker,
        )
        result = graph.invoke(
            {
                "messages": [],
                "question": "What is the capital of France?",
                "task_id": "lean-test",
                "attachment_path": "",
            },
            config={"recursion_limit": 12},
        )
        self.assertEqual(result["final_answer"], "Paris")
        self.assertEqual(result["verification_status"], "lean")

    def test_lean_tool_loop_is_forced_to_answer_after_two_rounds(self):
        def looping_worker(_):
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "calculate",
                        "args": {"expression": "2 + 2"},
                        "id": "lean-loop-call",
                        "type": "tool_call",
                    }
                ],
            )

        answer_inputs = []

        def answer_without_tools(messages):
            answer_inputs.append(messages)
            return AIMessage(content="4")

        worker = RunnableLambda(looping_worker)
        answerer = RunnableLambda(answer_without_tools)
        graph = build_lean_agent_graph(
            {
                "researcher": worker,
                "calculator": worker,
                "generalist": worker,
            },
            answerer,
        )
        result = graph.invoke(
            {
                "messages": [],
                "question": "Calculate the total.",
                "task_id": "lean-loop-test",
                "attachment_path": "",
            },
            config={"recursion_limit": 12},
        )
        self.assertEqual(result["tool_rounds"], 2)
        self.assertEqual(result["final_answer"], "4")
        self.assertEqual(len(answer_inputs[0]), 2)
        self.assertIsInstance(answer_inputs[0][1], HumanMessage)
        self.assertFalse(
            any(isinstance(message, ToolMessage) for message in answer_inputs[0])
        )
        self.assertIn("Calculate the total.", answer_inputs[0][1].content)
        self.assertIn("calculate result:", answer_inputs[0][1].content)

    def test_lean_research_tool_loop_allows_four_rounds(self):
        def looping_research(messages):
            completed_rounds = sum(
                isinstance(message, ToolMessage) for message in messages
            )
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "calculate",
                        "args": {"expression": "2 + 2"},
                        "id": f"research-loop-{completed_rounds}",
                        "type": "tool_call",
                    }
                ],
            )

        worker = RunnableLambda(looping_research)
        answerer = RunnableLambda(lambda _: AIMessage(content="quartz"))
        graph = build_lean_agent_graph(
            {
                "researcher": worker,
                "calculator": worker,
                "generalist": worker,
            },
            answerer,
        )
        result = graph.invoke(
            {
                "messages": [],
                "question": "What word was quoted in the journal article?",
                "task_id": "lean-research-loop-test",
                "attachment_path": "",
            },
            config={"recursion_limit": 16},
        )
        self.assertEqual(result["tool_rounds"], 4)
        self.assertEqual(result["final_answer"], "quartz")

    def test_lean_research_opens_publisher_after_search(self):
        search_result = ToolMessage(
            content=(
                "Source priority: third-party mirror\n"
                "Title: Mirror\n"
                "URL: https://mirror.example/paper\n"
                "Snippet: Copy\n\n"
                "Source priority: likely publisher\n"
                "Title: Original article\n"
                "URL: https://journal.example/article/123\n"
                "Snippet: Abstract"
            ),
            tool_call_id="search-call",
            name="search_web",
        )
        followup = _deterministic_research_followup(
            {
                "task_type": "research",
                "messages": [search_result],
                "tool_rounds": 1,
            }
        )
        self.assertIsNotNone(followup)
        self.assertEqual(followup.tool_calls[0]["name"], "fetch_webpage")
        self.assertEqual(
            followup.tool_calls[0]["args"]["url"],
            "https://journal.example/article/123",
        )

    def test_lean_research_prefers_relevant_article_over_editorial(self):
        question = (
            "In Dana Rivera's May 2018 article about clockwork birds, what word was "
            "quoted from two authors?"
        )
        search_call = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "search_web",
                    "args": {"query": "Dana Rivera May 2018 clockwork birds journal"},
                    "id": "search-call",
                    "type": "tool_call",
                }
            ],
        )
        search_result = ToolMessage(
            content=(
                "Source priority: general web\n"
                "Title: dana rivera may 2018 clockwork birds quoted word\n"
                "URL: https://www.etsy.com/market/dana_rivera_clockwork_birds\n"
                "Snippet: Dana Rivera May 2018 clockwork birds quoted authors.\n\n"
                "Source priority: general web\n"
                "Title: dana rivera may 2018 clockwork birds journal\n"
                "URL: https://unrelated.example/medical/"
                "vaccine-requirements-applicant-age.pdf\n"
                "Snippet: Dana Rivera wrote about clockwork birds in May 2018.\n\n"
                "Source priority: likely publisher\n"
                "Title: Editorial 2/2014\n"
                "URL: https://journal.example/articles/editorial-22014/\n"
                "Snippet: This issue includes an article about clockwork birds.\n\n"
                "Source priority: likely publisher\n"
                "Title: Clockwork Birds in Modern Children's Literature\n"
                "URL: https://journal.example/article/view/164228\n"
                "Snippet: Dana Rivera, published May 15, 2018."
            ),
            tool_call_id="search-call",
            name="search_web",
        )
        followup = _deterministic_research_followup(
            {
                "task_type": "research",
                "question": question,
                "messages": [search_call, search_result],
                "tool_rounds": 2,
            }
        )
        self.assertIsNotNone(followup)
        self.assertEqual(
            followup.tool_calls[0]["args"]["url"],
            "https://journal.example/article/view/164228",
        )

    def test_lean_research_tries_next_ranked_source_after_blocked_page(self):
        question = "In Dana Rivera's clockwork-bird article, what word was quoted?"
        search_call = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "search_web",
                    "args": {"query": "Dana Rivera clockwork bird quoted word"},
                    "id": "search-call",
                    "type": "tool_call",
                }
            ],
        )
        search_result = ToolMessage(
            content=(
                "Source priority: likely publisher\n"
                "Title: Clockwork Birds in Modern Literature\n"
                "URL: https://journal.example/article/view/42\n"
                "Snippet: Dana Rivera studies clockwork birds.\n\n"
                "Source priority: likely publisher\n"
                "Title: Clockwork Birds in Modern Literature (full text)\n"
                "URL: https://archive-journal.example/articles/clockwork-birds\n"
                "Snippet: Full publisher text by Dana Rivera."
            ),
            tool_call_id="search-call",
            name="search_web",
        )
        first_fetch = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "fetch_webpage",
                    "args": {"url": "https://journal.example/article/view/42"},
                    "id": "first-fetch",
                    "type": "tool_call",
                }
            ],
        )
        blocked = ToolMessage(
            content="BLOCKED PAGE: bot verification",
            tool_call_id="first-fetch",
            name="fetch_webpage",
        )
        followup = _deterministic_research_followup(
            {
                "task_type": "research",
                "question": question,
                "messages": [search_call, search_result, first_fetch, blocked],
                "tool_rounds": 2,
            }
        )
        self.assertIsNotNone(followup)
        self.assertEqual(
            followup.tool_calls[0]["args"]["url"],
            "https://archive-journal.example/articles/clockwork-birds",
        )

    def test_lean_research_reads_extensionless_download_as_pdf(self):
        search_result = ToolMessage(
            content=(
                "Source priority: likely publisher\n"
                "Title: Original PDF\n"
                "URL: https://journal.example/article/download/123/456\n"
                "Snippet: Abstract"
            ),
            tool_call_id="search-pdf",
            name="search_web",
        )
        followup = _deterministic_research_followup(
            {
                "task_type": "research",
                "messages": [search_result],
                "tool_rounds": 1,
            }
        )
        self.assertIsNotNone(followup)
        self.assertEqual(followup.tool_calls[0]["name"], "fetch_pdf")

    def test_tool_loop_is_bounded_and_recovers_calculator_result(self):
        def looping_worker(_):
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "calculate",
                        "args": {"expression": "2 + 2"},
                        "id": "loop-call",
                        "type": "tool_call",
                    }
                ],
            )

        worker = RunnableLambda(looping_worker)
        revising_verifier = RunnableLambda(
            lambda _: AIMessage(
                content="VERDICT: REVISE\nFEEDBACK: Recheck the expression."
            )
        )
        graph = build_agent_graph(
            {
                "researcher": worker,
                "calculator": worker,
                "generalist": worker,
            },
            revising_verifier,
        )
        result = graph.invoke(
            {
                "messages": [],
                "question": "Calculate the total.",
                "task_id": "loop-test",
                "attachment_path": "",
            },
            config={"recursion_limit": 40},
        )
        self.assertEqual(result["final_answer"], "4")
        self.assertEqual(result["verification_status"], "approved")

    def test_execution_trace_shows_actions_without_reasoning(self):
        messages = [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "calculate",
                        "args": {"expression": "2 + 2"},
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            ),
            ToolMessage(content="4", tool_call_id="call-1", name="calculate"),
            AIMessage(content="4"),
        ]
        trace = build_execution_trace(messages)
        rendered = format_execution_trace(trace)
        self.assertIn("TOOL CALL - calculate", rendered)
        self.assertIn("TOOL RESULT - calculate", rendered)
        self.assertIn("ANSWER CANDIDATE - assistant", rendered)
        self.assertNotIn("reasoning", rendered.lower())


if __name__ == "__main__":
    unittest.main()
