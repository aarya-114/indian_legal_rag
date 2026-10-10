import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


# This focused test exercises orchestration without loading models or retrieval
# dependencies. It does not replace the self-heal/relevance logic under test.
dotenv = types.ModuleType("dotenv")
dotenv.load_dotenv = lambda: None
sys.modules.setdefault("dotenv", dotenv)

openai = types.ModuleType("openai")
openai.OpenAI = Mock(return_value=Mock())
sys.modules.setdefault("openai", openai)
openai = sys.modules["openai"]

retrieval = types.ModuleType("src.retrieval.hybrid_retriever")
retrieval.HybridRetriever = object
retrieval.print_results = lambda *args, **kwargs: None
sys.modules.setdefault("src.retrieval.hybrid_retriever", retrieval)
os.environ["GROQ_API_KEY"] = "test-key"
os.environ["LLM_REQUEST_TIMEOUT_SECONDS"] = "37"
os.environ["LLM_MAX_OUTPUT_TOKENS"] = "900"
os.environ.setdefault("LLM_MODEL", "test-model")

from src.api import rag_pipeline
from src.evaluation.eval_runner import run_evaluation


def chunk(score):
    return {
        "text": "Test judgment",
        "metadata": {
            "judgment_title": "Test case", "court_type": "HC",
            "year": "2024", "case_type": "General", "doc_url": "",
        },
        "rerank_score": score,
    }


class SequenceRetriever:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.calls = 0
        self.requests = []

    def retrieve(self, **kwargs):
        self.calls += 1
        self.requests.append(kwargs)
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


class SelfHealingTelemetryTests(unittest.TestCase):
    def test_shared_groq_client_uses_configured_timeout(self):
        openai.OpenAI.assert_called_once_with(
            api_key="test-key",
            base_url="https://api.groq.com/openai/v1",
            timeout=37.0,
        )


    def run_query(self, retriever, rewrite):
        with (
            patch.object(rag_pipeline, "rewrite_query", side_effect=rewrite),
            patch.object(rag_pipeline, "generate_answer", return_value="Answer"),
            patch.object(rag_pipeline, "score_faithfulness", return_value=0.8),
        ):
            return rag_pipeline.rag_query("A legal question", retriever)

    def test_successful_retry_is_reported_as_recovery(self):
        retriever = SequenceRetriever([chunk(-1)], [chunk(2)])
        result = self.run_query(retriever, "rewritten question")

        self.assertEqual(retriever.calls, 2)
        self.assertEqual(result["scores"]["relevance"], 0.6)
        self.assertEqual(result["metadata"]["rewrite_attempted"], True)
        self.assertEqual(result["metadata"]["retry_attempted"], True)
        self.assertEqual(result["metadata"]["self_healed"], True)

    def test_answer_generation_accepts_complete_and_rejects_truncated_output(self):
        good_response = types.SimpleNamespace(
            id="answer-ok",
            choices=[types.SimpleNamespace(
                message=types.SimpleNamespace(content="Grounded legal answer", refusal=None),
                finish_reason="stop",
            )],
        )
        client = types.SimpleNamespace(
            chat=types.SimpleNamespace(
                completions=types.SimpleNamespace(create=Mock(return_value=good_response)),
            ),
        )
        source_chunk = chunk(1)
        with patch.object(rag_pipeline, "groq_client", client):
            answer = rag_pipeline.generate_answer(
                "A legal question", [source_chunk], {"detected_intent": "General"},
            )
        self.assertEqual(answer, "Grounded legal answer")
        request = client.chat.completions.create.call_args.kwargs
        self.assertEqual(request["model"], rag_pipeline.LLM_MODEL)
        self.assertEqual(rag_pipeline.LLM_MAX_OUTPUT_TOKENS, 900)
        self.assertEqual(
            request["max_completion_tokens"],
            rag_pipeline.LLM_MAX_OUTPUT_TOKENS,
        )
        self.assertEqual(request["reasoning_effort"], "low")
        self.assertIn("no more than 400 words", request["messages"][0]["content"])

        truncated_response = types.SimpleNamespace(
            id="answer-truncated",
            choices=[types.SimpleNamespace(
                message=types.SimpleNamespace(content="Partial answer", refusal=None),
                finish_reason="length",
            )],
        )
        client.chat.completions.create.return_value = truncated_response
        with patch.object(rag_pipeline, "groq_client", client):
            with self.assertRaises(rag_pipeline.LLMResponseError) as caught:
                rag_pipeline.generate_answer(
                    "A legal question", [source_chunk], {"detected_intent": "General"},
                )
        self.assertEqual(caught.exception.finish_reason, "length")

    def test_rag_query_preserves_truncation_as_generation_failure(self):
        retriever = SequenceRetriever([chunk(1)])
        truncated = rag_pipeline.LLMResponseError(
            "token limit reached", finish_reason="length",
        )
        with patch.object(rag_pipeline, "generate_answer", side_effect=truncated):
            with self.assertRaises(rag_pipeline.RAGGenerationError) as caught:
                rag_pipeline.rag_query("A legal question", retriever)

        self.assertEqual(caught.exception.stage, "Answer generation")
        self.assertEqual(caught.exception.cause.finish_reason, "length")
        self.assertEqual(caught.exception.relevance_score, 0.55)

    def test_rag_query_wraps_provider_timeout_as_generation_failure(self):
        retriever = SequenceRetriever([chunk(1)])
        client = types.SimpleNamespace(
            chat=types.SimpleNamespace(
                completions=types.SimpleNamespace(
                    create=Mock(side_effect=TimeoutError("provider timed out")),
                ),
            ),
        )
        with patch.object(rag_pipeline, "groq_client", client):
            with self.assertRaises(rag_pipeline.RAGGenerationError) as caught:
                rag_pipeline.rag_query("A legal question", retriever)

        self.assertEqual(caught.exception.stage, "Answer generation")
        self.assertIsInstance(caught.exception.cause, TimeoutError)
        self.assertFalse(caught.exception.retry_attempted)

    def test_rewrite_returns_short_assistant_query_with_bounded_request(self):
        response = types.SimpleNamespace(
                id="rewrite-ok",
                choices=[types.SimpleNamespace(
                    message=types.SimpleNamespace(
                    content="anticipatory bail", refusal=None,
                ),
                finish_reason="stop",
            )],
        )
        client = types.SimpleNamespace(
            chat=types.SimpleNamespace(
                completions=types.SimpleNamespace(create=Mock(return_value=response)),
            ),
        )
        with patch.object(rag_pipeline, "groq_client", client):
            rewritten = rag_pipeline.rewrite_query("anticipatory bail", "")

        self.assertEqual(rewritten, "anticipatory bail")
        request = client.chat.completions.create.call_args.kwargs
        self.assertEqual(request["model"], rag_pipeline.LLM_MODEL)
        self.assertEqual(request["max_completion_tokens"], 512)
        self.assertEqual(request["reasoning_effort"], "low")
        self.assertIn("at most 20 words", request["messages"][0]["content"])

    def test_unsupported_crpc_addition_is_rejected_and_falls_back_to_original(self):
        response = types.SimpleNamespace(
            id="rewrite-unsupported-statute",
            choices=[types.SimpleNamespace(
                message=types.SimpleNamespace(
                    content="CrPC anticipatory bail conditions", refusal=None,
                ),
                finish_reason="stop",
            )],
        )
        client = types.SimpleNamespace(
            chat=types.SimpleNamespace(
                completions=types.SimpleNamespace(create=Mock(return_value=response)),
            ),
        )
        retriever = SequenceRetriever([chunk(-1)], [chunk(2)])
        with (
            patch.object(rag_pipeline, "groq_client", client),
            patch.object(rag_pipeline, "generate_answer", return_value="Answer"),
            patch.object(rag_pipeline, "score_faithfulness", return_value=0.8),
        ):
            result = rag_pipeline.rag_query("anticipatory bail", retriever)

        with patch.object(rag_pipeline, "groq_client", client):
            with self.assertRaisesRegex(
                rag_pipeline.LLMResponseError, "not one safe, concise query",
            ):
                rag_pipeline.rewrite_query("anticipatory bail", "")

        self.assertEqual(retriever.requests[1]["query"], "anticipatory bail")
        self.assertIsNone(retriever.requests[1]["filter_case_type"])
        self.assertIsNone(retriever.requests[1]["filter_court_type"])
        self.assertFalse(result["metadata"]["rewrite_succeeded"])
        self.assertTrue(result["metadata"]["rewrite_failed"])
        self.assertTrue(result["metadata"]["self_healed"])

    def test_empty_or_truncated_rewrite_response_fails_explicitly(self):
        cases = [
            (None, "length", "truncated at the token limit"),
            ("partial query", "length", "truncated at the token limit"),
            ("   ", "stop", "contained no assistant text"),
        ]
        for content, finish_reason, message in cases:
            with self.subTest(content=content, finish_reason=finish_reason):
                response = types.SimpleNamespace(
                    id="rewrite-failed",
                    choices=[types.SimpleNamespace(
                        message=types.SimpleNamespace(content=content, refusal=None),
                        finish_reason=finish_reason,
                    )],
                )
                client = types.SimpleNamespace(
                    chat=types.SimpleNamespace(
                        completions=types.SimpleNamespace(
                            create=Mock(return_value=response),
                        ),
                    ),
                )
                with patch.object(rag_pipeline, "groq_client", client):
                    with self.assertRaisesRegex(
                        rag_pipeline.LLMResponseError, message,
                    ) as caught:
                        rag_pipeline.rewrite_query("anticipatory bail", "")
                self.assertNotIn("OpenRouter", str(caught.exception))

    def test_invalid_rewrite_is_rejected(self):
        for content in (
            "First query\nSecond query",
            "- Criminal dismissal query",
            "Query: employee dismissal",
            "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty one",
            "Section 438 employee dismissal",
            "Kesavananda Bharati v State of Kerala",
        ):
            with self.subTest(content=content):
                response = types.SimpleNamespace(
                    choices=[types.SimpleNamespace(
                        message=types.SimpleNamespace(content=content, refusal=None),
                        finish_reason="stop",
                    )],
                )
                client = types.SimpleNamespace(
                    chat=types.SimpleNamespace(
                        completions=types.SimpleNamespace(create=Mock(return_value=response)),
                    ),
                )
                with patch.object(rag_pipeline, "groq_client", client):
                    with self.assertRaises(rag_pipeline.LLMResponseError):
                        rag_pipeline.rewrite_query("employee dismissal", "")

    def test_failed_rewrite_falls_back_to_original_and_wider_retry(self):
        retriever = SequenceRetriever([chunk(-1)], [chunk(2)])
        result = self.run_query(retriever, RuntimeError("rewrite API unavailable"))

        self.assertEqual(retriever.calls, 2)
        self.assertEqual(retriever.requests[1]["query"], "A legal question")
        self.assertIsNone(retriever.requests[1]["filter_case_type"])
        self.assertIsNone(retriever.requests[1]["filter_court_type"])
        self.assertTrue(result["metadata"]["rewrite_attempted"])
        self.assertFalse(result["metadata"]["rewrite_succeeded"])
        self.assertTrue(result["metadata"]["rewrite_failed"])
        self.assertTrue(result["metadata"]["retry_succeeded"])
        self.assertTrue(result["metadata"]["self_healed"])

    def test_rewrite_failure_with_explicit_case_filter_preserves_caller_filter(self):
        retriever = SequenceRetriever([chunk(-1)], [chunk(2)])
        with (
            patch.object(rag_pipeline, "rewrite_query", side_effect=RuntimeError("API")),
            patch.object(rag_pipeline, "generate_answer", return_value="Answer"),
            patch.object(rag_pipeline, "score_faithfulness", return_value=0.8),
        ):
            result = rag_pipeline.rag_query(
                "A legal question", retriever, filter_case_type="Criminal",
            )

        self.assertEqual(retriever.requests[1]["query"], "A legal question")
        self.assertEqual(retriever.requests[1]["filter_case_type"], "Criminal")
        self.assertIsNone(retriever.requests[1]["filter_court_type"])
        self.assertTrue(result["metadata"]["self_healed"])

    def test_failed_rewrite_and_failed_retry_preserve_attempt_state(self):
        rewrite_failure = SequenceRetriever([chunk(-1)])
        with self.assertRaises(rag_pipeline.RAGGenerationError) as caught:
            self.run_query(rewrite_failure, RuntimeError("rewrite failed"))
        self.assertTrue(caught.exception.rewrite_attempted)
        self.assertTrue(caught.exception.retry_attempted)
        self.assertFalse(caught.exception.self_healed)
        self.assertTrue(caught.exception.rewrite_failed)
        self.assertTrue(caught.exception.retry_failed)

        retry_failure = SequenceRetriever([chunk(-1)], RuntimeError("retry failed"))
        with self.assertRaises(rag_pipeline.RAGGenerationError) as caught:
            self.run_query(retry_failure, "rewritten question")
        self.assertTrue(caught.exception.rewrite_attempted)
        self.assertTrue(caught.exception.retry_attempted)
        self.assertFalse(caught.exception.self_healed)
        self.assertTrue(caught.exception.retry_failed)

        unrecovered = SequenceRetriever([chunk(-1)], [chunk(0)])
        result = self.run_query(unrecovered, "rewritten question")
        self.assertTrue(result["metadata"]["retry_attempted"])
        self.assertFalse(result["metadata"]["self_healed"])

    def test_question_above_threshold_does_not_trigger_self_healing(self):
        retriever = SequenceRetriever([chunk(1)])
        result = self.run_query(retriever, AssertionError("rewrite must not run"))

        self.assertEqual(retriever.calls, 1)
        self.assertFalse(result["metadata"]["rewrite_attempted"])
        self.assertFalse(result["metadata"]["retry_attempted"])
        self.assertFalse(result["metadata"]["self_healed"])

    def test_faithfulness_zero_means_no_extracted_reference_matched(self):
        source = [{"text": "The court considered Section 12 of the relevant Act."}]
        self.assertEqual(
            rag_pipeline.score_faithfulness(
                "The requested power is under Section 11A.", source,
            ),
            0.0,
        )
        self.assertEqual(
            rag_pipeline.score_faithfulness("A general explanation.", source),
            0.7,
        )

    def test_evaluation_counts_attempts_separately_from_recoveries(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "eval.json"
            output = Path(directory) / "results.json"
            dataset.write_text(json.dumps([
                {"question": "Failed recovery", "ground_truth": "answer"},
                {"question": "Recovered", "ground_truth": "answer"},
                {"question": "Rewrite error", "ground_truth": "answer"},
                {"question": "Low relevance and generation error", "ground_truth": "answer"},
                {"question": "Fallback recovered", "ground_truth": "answer"},
            ]), encoding="utf-8")

            def query(question, retriever, top_k):
                if question == "Rewrite error":
                    raise rag_pipeline.RAGGenerationError(
                        "Query rewrite", 0.4, False, RuntimeError("rewrite failed"),
                        rewrite_attempted=True, retry_attempted=False,
                        rewrite_failed=True,
                    )
                if question == "Low relevance and generation error":
                    raise rag_pipeline.RAGGenerationError(
                        "Answer generation", 0.468, False,
                        RuntimeError("generation failed"),
                        rewrite_attempted=True, retry_attempted=True,
                        rewrite_succeeded=True, retry_succeeded=True,
                    )
                if question == "Fallback recovered":
                    return {
                        "answer": "answer",
                        "scores": {"relevance": 0.6, "faithfulness": 0.8},
                        "metadata": {
                            "self_healed": True,
                            "rewrite_attempted": True,
                            "rewrite_succeeded": False,
                            "rewrite_failed": True,
                            "retry_attempted": True,
                            "retry_succeeded": True,
                            "retry_failed": False,
                        },
                        "sources": [],
                    }
                recovered = question == "Recovered"
                return {
                    "answer": "answer",
                    "scores": {"relevance": 0.6, "faithfulness": 0.8},
                    "metadata": {
                        "self_healed": recovered,
                        "rewrite_attempted": True,
                        "rewrite_succeeded": recovered,
                        "rewrite_failed": not recovered,
                        "retry_attempted": True,
                        "retry_succeeded": True,
                    },
                    "sources": [],
                }

            with patch("src.evaluation.eval_runner.time.sleep"):
                summary = run_evaluation(
                    str(dataset), str(output), retriever=object(), query_fn=query,
                )
            records = json.loads(output.read_text(encoding="utf-8"))["results"]

        self.assertEqual(summary["rewrite_attempt_count"], 5)
        self.assertEqual(summary["rewrite_success_count"], 2)
        self.assertEqual(summary["rewrite_failed_count"], 3)
        self.assertEqual(summary["retry_attempt_count"], 4)
        self.assertEqual(summary["retry_success_count"], 4)
        self.assertEqual(summary["retry_failed_count"], 0)
        self.assertEqual(summary["self_heal_count"], 2)
        self.assertEqual(summary["self_heal_questions"], 5)
        self.assertEqual(summary["self_heal_eligible_questions"], 5)
        self.assertEqual(summary["successful_recovery_count"], 2)
        self.assertEqual(summary["failed_recovery_count"], 3)
        self.assertEqual(summary["self_heal_rate"], 0.4)
        self.assertEqual(records[3]["self_heal_status"], "failed")
        self.assertFalse(records[3]["self_healed"])
        self.assertEqual(records[4]["self_heal_status"], "recovered")
        self.assertFalse(records[4]["rewrite_succeeded"])
        self.assertTrue(records[4]["rewrite_failed"])

    def test_evaluation_persists_reference_fallback_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "eval.json"
            output = Path(directory) / "results.json"
            dataset.write_text(json.dumps([{
                "question": "Section 11A powers", "ground_truth": "answer",
            }]), encoding="utf-8")
            query = lambda *args, **kwargs: {
                "answer": "answer",
                "scores": {"relevance": 0.7, "faithfulness": 0.8},
                "metadata": {
                    "self_healed": True,
                    "fallback_activated": True,
                    "reference_recovered": True,
                    "retry_attempted": True,
                    "retry_succeeded": True,
                    "initial_chunk_ids": ["initial"],
                    "fallback_candidate_ids": ["candidate"],
                    "fallback_chunk_ids": ["recovered"],
                    "final_chunk_ids": ["recovered"],
                },
                "sources": [],
            }
            with patch("src.evaluation.eval_runner.time.sleep"):
                summary = run_evaluation(
                    str(dataset), str(output), retriever=object(), query_fn=query,
                )
            record = json.loads(output.read_text(encoding="utf-8"))["results"][0]

        self.assertEqual(summary["retry_attempt_count"], 1)
        self.assertEqual(summary["successful_recovery_count"], 1)
        self.assertEqual(record["initial_chunk_ids"], ["initial"])
        self.assertEqual(record["fallback_candidate_ids"], ["candidate"])
        self.assertEqual(record["fallback_chunk_ids"], ["recovered"])
        self.assertEqual(record["final_chunk_ids"], ["recovered"])
        self.assertTrue(record["fallback_activated"])
        self.assertTrue(record["reference_recovered"])


if __name__ == "__main__":
    unittest.main()


class CitationIntegrityTests(unittest.TestCase):
    @staticmethod
    def chunk(chunk_id, text, url, title="Same title"):
        return {
            "text": text,
            "metadata": {
                "chunk_id": chunk_id,
                "doc_url": url,
                "judgment_title": title,
                "court_type": "HC",
                "year": "2024",
                "case_type": "General",
            },
            "rerank_score": 0.75,
        }

    def test_valid_source_reference_maps_to_returned_judgment(self):
        chunks = [self.chunk("judgment_1_chunk_0001", "Evidence A", "https://court/a")]
        sources = rag_pipeline.build_judgment_sources(chunks)
        response_sources = [{key: value for key, value in sources[0].items()
                             if key not in {"passages", "chunk_ids"}}]

        self.assertEqual(sources[0]["source_id"], "1")
        self.assertEqual(
            rag_pipeline.validate_source_references("Supported [Source 1]", response_sources),
            {"referenced_source_ids": ["1"], "unknown_source_ids": []},
        )

    def test_rag_result_uses_same_ids_in_answer_validation_and_sources(self):
        retrieved = self.chunk(
            "judgment_1_chunk_0001", "Evidence A", "https://court/a",
        )
        with (
            patch.object(rag_pipeline, "score_relevance", return_value=0.8),
            patch.object(rag_pipeline, "generate_answer", return_value="Answer [Source 1]"),
            patch.object(rag_pipeline, "score_faithfulness", return_value=0.5),
        ):
            result = rag_pipeline.rag_query("Question", SequenceRetriever([retrieved]))

        self.assertEqual(result["sources"][0]["source_id"], "1")
        self.assertEqual(
            result["citation_validation"],
            {"referenced_source_ids": ["1"], "unknown_source_ids": []},
        )
        self.assertIn("Answer [Source 1]", result["answer"])

    def test_rag_result_warns_for_generated_unknown_source_id(self):
        retrieved = self.chunk(
            "judgment_1_chunk_0001", "Evidence A", "https://court/a",
        )
        with (
            patch.object(rag_pipeline, "score_relevance", return_value=0.8),
            patch.object(rag_pipeline, "generate_answer", return_value="Answer [Source 5]"),
            patch.object(rag_pipeline, "score_faithfulness", return_value=0.8),
        ):
            result = rag_pipeline.rag_query("Question", SequenceRetriever([retrieved]))

        self.assertEqual(result["citation_validation"]["unknown_source_ids"], ["5"])
        self.assertIn("unknown source ID(s): 5", result["warning"])

    def test_unknown_source_reference_is_reported_not_remapped(self):
        sources = [{"source_id": "1"}]
        self.assertEqual(
            rag_pipeline.validate_source_references("Claim [Source 2]", sources),
            {"referenced_source_ids": ["2"], "unknown_source_ids": ["2"]},
        )

    def test_multiple_chunks_for_judgment_share_prompt_and_response_id(self):
        chunks = [
            self.chunk("judgment_1_chunk_0001", "First passage", "https://court/a"),
            self.chunk("judgment_1_chunk_0002", "Second passage", "https://court/a"),
            self.chunk("judgment_2_chunk_0001", "Other judgment", "https://court/b"),
        ]
        groups = rag_pipeline.build_judgment_sources(chunks)
        self.assertEqual([group["source_id"] for group in groups], ["1", "2"])
        self.assertEqual(groups[0]["passages"], ["First passage", "Second passage"])
        self.assertEqual(groups[0]["chunk_ids"], [
            "judgment_1_chunk_0001", "judgment_1_chunk_0002",
        ])

        response = types.SimpleNamespace(choices=[types.SimpleNamespace(
            message=types.SimpleNamespace(content="Answer [Source 1]", refusal=None),
            finish_reason="stop",
        )])
        client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(
            create=Mock(return_value=response),
        )))
        with patch.object(rag_pipeline, "groq_client", client):
            rag_pipeline.generate_answer("Question", chunks, {})
        prompt = client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
        self.assertEqual(prompt.count("[Source 1:"), 1)
        self.assertIn("First passage", prompt)
        self.assertIn("Second passage", prompt)
        self.assertIn("[Source 2:", prompt)

    def test_missing_or_empty_metadata_does_not_merge_by_title_or_invent_link(self):
        chunks = [
            {"text": "Passage one", "metadata": {
                "chunk_id": "judgment_1_chunk_0001", "judgment_title": "",
            }},
            {"text": "Passage two", "metadata": {
                "chunk_id": "judgment_2_chunk_0001",
            }},
        ]
        groups = rag_pipeline.build_judgment_sources(chunks)
        self.assertEqual(len(groups), 2)
        self.assertEqual([group["source_id"] for group in groups], ["1", "2"])
        self.assertEqual([group["url"] for group in groups], ["", ""])
        self.assertEqual([group["title"] for group in groups], ["", ""])
        self.assertEqual(
            rag_pipeline.build_judgment_sources([{"text": "   ", "metadata": {}}]),
            [],
        )
        self.assertEqual(
            rag_pipeline.build_judgment_sources([{"text": "No identity", "metadata": {}}]),
            [],
        )
        self.assertEqual(
            rag_pipeline.validate_source_references("Claim [Source 1]", []),
            {"referenced_source_ids": ["1"], "unknown_source_ids": ["1"]},
        )

    def test_valid_citation_id_does_not_mark_unsupported_claim_verified(self):
        answer = "Section 99 of Act 2099 permits dismissal [Source 1]."
        sources = [{"source_id": "1"}]
        validation = rag_pipeline.validate_source_references(answer, sources)
        faithfulness = rag_pipeline.score_faithfulness(
            answer, [{"text": "A passage with no such provision."}],
        )
        self.assertEqual(validation["unknown_source_ids"], [])
        self.assertEqual(faithfulness, 0.0)
        self.assertNotIn("legally_verified", validation)
