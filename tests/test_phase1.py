import asyncio
import os
import subprocess
import sys
import types
import unittest
from unittest.mock import Mock, patch

# Keep model and vector database initialization out of unit tests.
chromadb = types.ModuleType("chromadb")
chromadb.PersistentClient = Mock()
config = types.ModuleType("chromadb.config")
config.Settings = Mock
sys.modules.setdefault("chromadb", chromadb)
sys.modules.setdefault("chromadb.config", config)
transformers = types.ModuleType("sentence_transformers")
transformers.SentenceTransformer = Mock(return_value=Mock())
transformers.CrossEncoder = Mock(return_value=Mock())
sys.modules.setdefault("sentence_transformers", transformers)
bm25 = types.ModuleType("rank_bm25")
bm25.BM25Okapi = Mock
sys.modules.setdefault("rank_bm25", bm25)
# PostgreSQL is optional for these unit tests. Stub its import-time surface so
# Windows native-driver policy does not block tests that exercise other code.
psycopg2 = types.ModuleType("psycopg2")
psycopg2.connect = Mock()
psycopg2_extras = types.ModuleType("psycopg2.extras")
psycopg2_extras.RealDictCursor = object
psycopg2.extras = psycopg2_extras
sys.modules.setdefault("psycopg2", psycopg2)
sys.modules.setdefault("psycopg2.extras", psycopg2_extras)
openai = types.ModuleType("openai")
openai.OpenAI = Mock(return_value=Mock())
sys.modules.setdefault("openai", openai)
os.environ["GROQ_API_KEY"] = "test-key"
os.environ["LLM_REQUEST_TIMEOUT_SECONDS"] = "37"
os.environ["LLM_MAX_OUTPUT_TOKENS"] = "1024"
# The provider is mocked in tests; this sentinel only satisfies config validation.
os.environ.setdefault("LLM_MODEL", "test")

from src.api import main, rag_pipeline
from src.chunking.legal_chunker import chunk_all_judgments, parse_args
from src.retrieval.hybrid_retriever import HybridRetriever
from src.evaluation.eval_runner import compute_token_overlap
from src.evaluation.score_gate import check_gate
from src.evaluation.eval_runner import run_evaluation
from src.embeddings.embedder import verify_collection_count


class PhaseOneTests(unittest.TestCase):
    @staticmethod
    def retrieval_chunk(chunk_id, text, case_type="General"):
        return {
            "text": text,
            "metadata": {"chunk_id": chunk_id, "case_type": case_type},
            "dense_score": 0.8,
            "sparse_score": 1.0,
            "rerank_score": 4.0,
        }

    def test_explicit_section_reference_normalization(self):
        self.assertEqual(
            rag_pipeline.extract_statutory_section_reference(
                "Power under Section 11-A of the Industrial Disputes Act?"
            ),
            "section 11a",
        )
        self.assertIsNone(
            rag_pipeline.extract_statutory_section_reference("Industrial disputes")
        )

    def test_reference_fallback_merges_candidates_before_reranking(self):
        retriever = HybridRetriever.__new__(HybridRetriever)
        initial = [self.retrieval_chunk(f"initial-{i}", "Industrial dismissal")
                   for i in range(8)]
        target = self.retrieval_chunk(
            "mislabelled-section", "Section 11A permits the Tribunal to alter punishment.",
            "Tax",
        )
        retriever._retrieve_candidates = Mock(side_effect=[
            (initial, initial), ([target], [target]),
        ])
        rerank_inputs = []

        def rerank(query, candidates, top_k=5):
            rerank_inputs.append(list(candidates))
            ordered = sorted(
                candidates,
                key=lambda item: item["metadata"]["chunk_id"] == "mislabelled-section",
                reverse=True,
            )
            return ordered[:top_k]

        retriever.rerank = rerank
        results, diagnostics = retriever.retrieve_with_reference_fallback(
            "Section 11A power", "section 11a", top_k=5,
            filter_case_type="Industrial&Labour", fallback_case_type=None,
        )
        self.assertTrue(diagnostics["fallback_activated"])
        self.assertTrue(diagnostics["reference_recovered"])
        self.assertIsNone(retriever._retrieve_candidates.call_args_list[1].args[1])
        self.assertEqual(results[0]["metadata"]["chunk_id"], "mislabelled-section")
        self.assertIn(target["metadata"]["chunk_id"], [
            item["metadata"]["chunk_id"] for item in rerank_inputs[-1]
        ])
        self.assertGreater(len(rerank_inputs[-1]), 5)
        self.assertEqual(len(results), 5)
        self.assertEqual(retriever._retrieve_candidates.call_count, 2)

    def test_existing_section_reference_skips_fallback(self):
        retriever = HybridRetriever.__new__(HybridRetriever)
        section = self.retrieval_chunk("section", "Section 11-A modifies punishment")
        retriever._retrieve_candidates = Mock(return_value=([section], [section]))
        retriever._fuse_and_rerank = Mock(return_value=[section])
        results, diagnostics = retriever.retrieve_with_reference_fallback(
            "Section 11A powers", "section 11a",
        )
        self.assertEqual(results, [section])
        self.assertFalse(diagnostics["fallback_activated"])
        retriever._retrieve_candidates.assert_called_once()

    def test_reference_fallback_failure_returns_initial_results(self):
        retriever = HybridRetriever.__new__(HybridRetriever)
        initial = self.retrieval_chunk("initial", "Industrial dismissal")
        retriever._retrieve_candidates = Mock(side_effect=[
            ([initial], [initial]), RuntimeError("fallback unavailable"),
        ])
        retriever._fuse_and_rerank = Mock(return_value=[initial])
        results, diagnostics = retriever.retrieve_with_reference_fallback(
            "Section 11A powers", "section 11a",
        )
        self.assertEqual(results, [initial])
        self.assertTrue(diagnostics["fallback_activated"])
        self.assertFalse(diagnostics["reference_recovered"])
        self.assertEqual(diagnostics["fallback_error"], "fallback unavailable")

    def test_high_relevance_missing_reference_uses_one_fallback_and_preserves_filter(self):
        class FallbackRetriever:
            def __init__(self):
                self.calls = []

            def retrieve_with_reference_fallback(self, **kwargs):
                self.calls.append(kwargs)
                initial = PhaseOneTests.retrieval_chunk("initial", "Related industrial matter")
                recovered = PhaseOneTests.retrieval_chunk(
                    "found-section", "Section 11A permits alteration of punishment.",
                    "Industrial&Labour",
                )
                return [recovered], {
                    "initial_chunk_ids": ["initial"],
                    "fallback_candidate_ids": ["found-section"],
                    "fallback_chunk_ids": ["found-section"],
                    "final_chunk_ids": ["found-section"],
                    "fallback_activated": True,
                    "reference_recovered": True,
                    "fallback_error": None,
                }

        retriever = FallbackRetriever()
        with (
            patch.object(rag_pipeline, "generate_answer", return_value="Grounded answer"),
            patch.object(rag_pipeline, "score_faithfulness", return_value=0.8),
            patch.object(rag_pipeline, "rewrite_query", side_effect=AssertionError("duplicate recovery")),
        ):
            result = rag_pipeline.rag_query(
                "What is the power under Section 11A of Industrial Disputes Act?",
                retriever,
                filter_case_type="Industrial&Labour",
            )
        self.assertEqual(len(retriever.calls), 1)
        self.assertEqual(retriever.calls[0]["filter_case_type"], "Industrial&Labour")
        self.assertEqual(retriever.calls[0]["fallback_case_type"], "Industrial&Labour")
        # The explicit filter remains in force during both passes.
        self.assertTrue(result["metadata"]["fallback_activated"])
        self.assertTrue(result["metadata"]["reference_recovered"])
        self.assertEqual(result["scores"]["relevance"], 0.7)
        self.assertTrue(result["metadata"]["retry_attempted"])
        self.assertTrue(result["metadata"]["self_healed"])
        self.assertEqual(result["metadata"]["initial_chunk_ids"], ["initial"])
        self.assertEqual(result["metadata"]["final_chunk_ids"], ["found-section"])

    def test_groq_compatible_client_uses_environment_key_and_endpoint_once(self):
        openai.OpenAI.assert_called_once_with(
            api_key="test-key",
            base_url="https://api.groq.com/openai/v1",
            timeout=37.0,
        )

    def test_openrouter_null_content_is_reported_without_strip_error(self):
        response = types.SimpleNamespace(
            id="response-test",
            choices=[types.SimpleNamespace(
                message=types.SimpleNamespace(content=None, refusal=None),
                finish_reason="stop",
            )],
        )
        with self.assertRaisesRegex(
            rag_pipeline.OpenRouterResponseError,
            "answer generation response contained no assistant text",
        ):
            rag_pipeline.extract_response_text(response, "answer generation")

    def test_eval_keeps_relevance_when_generation_fails_and_q5_succeeds(self):
        import json
        import tempfile
        from pathlib import Path

        dataset = [
            {"question": "Q1", "ground_truth": "bail conditions"},
            {"question": "Q5", "ground_truth": "bail condition"},
        ]

        def query(question, retriever, top_k):
            if question == "Q1":
                cause = rag_pipeline.OpenRouterResponseError("no assistant text")
                raise rag_pipeline.RAGGenerationError(
                    "Answer generation", 0.6, False, cause
                )
            return {
                "answer": "bail condition",
                "scores": {"relevance": 0.8, "faithfulness": 0.9},
                "metadata": {"self_healed": False, "detected_intent": "Criminal"},
                "sources": [],
            }

        with tempfile.TemporaryDirectory() as directory:
            data_path = Path(directory) / "eval.json"
            results_path = Path(directory) / "results.json"
            data_path.write_text(json.dumps(dataset), encoding="utf-8")
            with patch("src.evaluation.eval_runner.time.sleep"):
                summary = run_evaluation(
                    str(data_path), str(results_path),
                    retriever=object(), query_fn=query,
                )
            results = json.loads(results_path.read_text(encoding="utf-8"))["results"]

        self.assertEqual(results[0]["scores"], {
            "relevance": 0.6, "faithfulness": None, "answer_overlap": None,
        })
        self.assertEqual(results[1]["scores"]["answer_overlap"], 1.0)
        self.assertEqual(summary["avg_relevance"], 0.7)
        self.assertEqual(summary["avg_faithfulness"], 0.9)
        self.assertEqual(summary["avg_answer_overlap"], 1.0)
        self.assertEqual(summary["metric_counts"], {
            "relevance": 2, "faithfulness": 1, "answer_overlap": 1,
        })

    def test_eval_marks_missing_reference_answer_unavailable(self):
        import json
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            data_path = Path(directory) / "eval.json"
            results_path = Path(directory) / "results.json"
            data_path.write_text(json.dumps([{
                "question": "Question", "ground_truth": None,
            }]), encoding="utf-8")
            query = Mock(return_value={
                "answer": "An answer",
                "scores": {"relevance": 0.8, "faithfulness": 0.9},
                "metadata": {"self_healed": False}, "sources": [],
            })
            with patch("src.evaluation.eval_runner.time.sleep"):
                summary = run_evaluation(
                    str(data_path), str(results_path),
                    retriever=object(), query_fn=query,
                )
            result = json.loads(results_path.read_text(encoding="utf-8"))["results"][0]

        self.assertIsNone(result["scores"]["answer_overlap"])
        self.assertEqual(
            result["metric_status"]["answer_overlap"],
            "unavailable: reference answer missing",
        )
        self.assertIsNone(summary["avg_answer_overlap"])

    def test_embedder_rejects_collection_count_mismatch(self):
        collection = Mock()
        collection.count.return_value = 4

        with self.assertLogs("src.embeddings.embedder", level="INFO") as logs:
            with self.assertRaisesRegex(RuntimeError, "expected 5, found 4"):
                verify_collection_count(collection, expected_count=5)

        self.assertTrue(any("Expected chunk count: 5" in line for line in logs.output))
        self.assertTrue(any("Actual Chroma collection count: 4" in line for line in logs.output))

    def test_evaluation_cli_help_exits_without_loading_retriever(self):
        result = subprocess.run(
            [sys.executable, "src/evaluation/eval_runner.py", "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--dataset", result.stdout)
        self.assertIn("--output", result.stdout)
        self.assertIn("data/eval/eval_dataset.json", result.stdout)
        self.assertIn("data/eval/eval_results.json", result.stdout)
        self.assertNotIn("Loading embedding model", result.stdout + result.stderr)

    def test_chunker_defaults_to_full_dataset_and_supports_limit(self):
        import inspect
        self.assertIsNone(inspect.signature(chunk_all_judgments).parameters["max_judgments"].default)
        self.assertIsNone(parse_args([]).limit)
        self.assertEqual(parse_args(["--limit", "3"]).limit, 3)

    def test_sparse_retrieval_applies_case_type_filter(self):
        retriever = HybridRetriever.__new__(HybridRetriever)
        base = {
            "text": "bail bail", "court_type": "HC", "year": "2020",
            "judgment_title": "Test judgment", "court_name_normalized": "HC",
            "petitioner": "A", "respondent": "B", "doc_url": "",
            "cites_count": 0, "cited_by_count": 0,
        }
        retriever.all_chunks = [
            {**base, "chunk_id": "a", "case_type": "Criminal"},
            {**base, "chunk_id": "b", "case_type": "Tax"},
        ]
        retriever.bm25 = Mock()
        retriever.bm25.get_scores.return_value = [1.0, 10.0]
        results = retriever.sparse_search("bail", n_results=1, filter_case_type="Criminal")
        self.assertEqual([item["metadata"]["case_type"] for item in results], ["Criminal"])

    def test_rag_pipeline_passes_requested_filter_to_retriever(self):
        chunk = {"text": "A judgment", "metadata": {"judgment_title": "Case", "court_type": "HC", "year": "2020", "case_type": "Tax", "doc_url": ""}, "rerank_score": 1.0}
        retriever = Mock()
        retriever.retrieve.return_value = [chunk]
        response = types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="Grounded answer"))])
        with patch.object(
            rag_pipeline.groq_client.chat.completions,
            "create",
            return_value=response,
        ) as create_call:
            rag_pipeline.rag_query("Explain the applicable legal test", retriever, filter_case_type="Tax")
        self.assertEqual(retriever.retrieve.call_args.kwargs["filter_case_type"], "Tax")
        self.assertEqual(
            create_call.call_args.kwargs["model"],
            os.environ["LLM_MODEL"],
        )

    def test_api_forwards_filter_case_type(self):
        request = main.QueryRequest(query="Explain anticipatory bail conditions", filter_case_type="Criminal")
        main.retriever = object()
        result = {
            "query": request.query,
            "answer": "Answer [Source 1]",
            "sources": [{
                "source_id": "1", "title": "Example case", "court": "HC",
                "year": "2024", "case_type": "Criminal", "url": "",
                "relevance_score": 0.8,
            }],
            "citation_validation": {
                "referenced_source_ids": ["1"], "unknown_source_ids": [],
            },
            "scores": {"relevance": 0.8, "faithfulness": 0.8},
            "metadata": {"detected_intent": "Criminal", "self_healed": False, "chunks_retrieved": 1, "model": "test"},
        }
        with patch("src.api.rag_pipeline.rag_query", return_value=result) as query:
            response = asyncio.run(main.query_endpoint(request))
        self.assertEqual(query.call_args.kwargs["filter_case_type"], "Criminal")
        self.assertEqual(response.sources[0].source_id, "1")
        self.assertEqual(response.citation_validation.unknown_source_ids, [])
        main.retriever = None

    def test_query_request_rejects_missing_empty_and_whitespace_queries(self):
        for payload in ({}, {"query": ""}, {"query": "           "}):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                main.QueryRequest(**payload)

    def test_query_request_rejects_wrong_type_and_query_over_4000_characters(self):
        invalid_payloads = (
            {"query": ["not", "a", "string"]},
            {"query": "x" * 4001},
        )
        for payload in invalid_payloads:
            with self.subTest(payload_type=type(payload["query"])), self.assertRaises(ValueError):
                main.QueryRequest(**payload)

    def test_query_request_accepts_query_at_4000_character_limit(self):
        request = main.QueryRequest(query="x" * 4000)
        self.assertEqual(len(request.query), 4000)

    def test_query_body_limit_returns_413_for_declared_oversized_body(self):
        async def exercise():
            sent = []

            async def receive():
                return {"type": "http.request", "body": b"", "more_body": False}

            async def send(message):
                sent.append(message)

            async def app(scope, receive, send):
                raise AssertionError("oversized body must be rejected before route")

            middleware = main.RequestBodyLimitMiddleware(app, max_body_bytes=8)
            scope = {
                "type": "http", "method": "POST", "path": "/query",
                "headers": [(b"content-length", b"9")],
            }
            await middleware(scope, receive, send)
            return sent

        sent = asyncio.run(exercise())
        self.assertEqual(sent[0]["status"], 413)
        self.assertIn(b"8-byte limit", sent[1]["body"])

    def test_query_body_limit_counts_streamed_body_without_content_length(self):
        async def exercise():
            sent = []
            messages = iter((
                {"type": "http.request", "body": b"123456", "more_body": True},
                {"type": "http.request", "body": b"789", "more_body": False},
            ))

            async def receive():
                return next(messages)

            async def send(message):
                sent.append(message)

            async def app(scope, receive, send):
                raise AssertionError("oversized body must be rejected before route")

            middleware = main.RequestBodyLimitMiddleware(app, max_body_bytes=8)
            scope = {"type": "http", "method": "POST", "path": "/query", "headers": []}
            await middleware(scope, receive, send)
            return sent

        sent = asyncio.run(exercise())
        self.assertEqual(sent[0]["status"], 413)

    def test_query_rate_limit_allows_requests_within_limit(self):
        middleware = main.QueryRateLimitMiddleware
        async def exercise():
            calls = []
            async def app(scope, receive, send):
                calls.append(scope["path"])
            limiter = middleware(app, max_requests=2, window_seconds=60)
            for _ in range(2):
                await limiter(
                    {"type": "http", "method": "POST", "path": "/query", "client": ("10.0.0.1", 1)},
                    None, None,
                )
            return calls
        self.assertEqual(asyncio.run(exercise()), ["/query", "/query"])

    def test_query_rate_limit_returns_safe_429_after_limit(self):
        async def exercise():
            calls = []
            responses = []
            async def app(scope, receive, send):
                calls.append(scope["path"])
            async def send(message):
                responses.append(message)
            limiter = main.QueryRateLimitMiddleware(app, max_requests=1, window_seconds=60)
            scope = {"type": "http", "method": "POST", "path": "/query", "client": ("10.0.0.1", 1)}
            await limiter(scope, None, send)
            await limiter(scope, None, send)
            return calls, responses
        calls, responses = asyncio.run(exercise())
        self.assertEqual(len(calls), 1)
        self.assertEqual(responses[0]["status"], 429)
        headers = dict(responses[0]["headers"])
        self.assertIn(b"retry-after", headers)
        self.assertEqual(responses[-1]["body"], b'{"detail":"Rate limit exceeded. Please retry later."}')

    def test_query_rate_limit_is_independent_per_client_and_only_targets_query(self):
        async def exercise():
            calls = []
            responses = []
            async def app(scope, receive, send):
                calls.append((scope["client"][0], scope["path"]))
            async def send(message):
                responses.append(message)
            limiter = main.QueryRateLimitMiddleware(app, max_requests=1, window_seconds=60)
            async def request(ip, path="/query"):
                await limiter({
                    "type": "http", "method": "POST", "path": path,
                    "client": (ip, 1),
                }, None, send)
            await request("10.0.0.1")
            await request("10.0.0.1")
            await request("10.0.0.2")
            await request("10.0.0.1", "/health")
            return calls, responses
        calls, responses = asyncio.run(exercise())
        self.assertEqual(calls, [
            ("10.0.0.1", "/query"),
            ("10.0.0.2", "/query"),
            ("10.0.0.1", "/health"),
        ])
        self.assertEqual(responses[0]["status"], 429)

    def test_query_response_survives_optional_logging_failure(self):
        result = {
            "query": "Explain anticipatory bail conditions",
            "answer": "A grounded answer",
            "sources": [],
            "scores": {"relevance": 0.8, "faithfulness": 0.9},
            "metadata": {
                "detected_intent": "Criminal", "self_healed": False,
                "chunks_retrieved": 1, "model": "test-model",
            },
            "warning": None,
        }
        main.retriever = object()
        with (
            patch("src.api.rag_pipeline.rag_query", return_value=result),
            patch.object(main, "log_to_postgres", side_effect=RuntimeError("password=secret")),
        ):
            response = asyncio.run(main.query_endpoint(
                main.QueryRequest(query="Explain anticipatory bail conditions")
            ))
        main.retriever = None
        self.assertEqual(response.answer, "A grounded answer")

    def test_query_errors_are_safe_and_provider_failure_is_bad_gateway(self):
        main.retriever = object()
        provider_error = rag_pipeline.RAGGenerationError(
            "Answer generation", 0.7, False, TimeoutError("credential=secret"),
        )
        with patch("src.api.rag_pipeline.rag_query", side_effect=provider_error):
            with self.assertRaises(main.HTTPException) as caught:
                asyncio.run(main.query_endpoint(
                    main.QueryRequest(query="Explain anticipatory bail conditions")
                ))
        self.assertEqual(caught.exception.status_code, 502)
        self.assertNotIn("secret", caught.exception.detail)

        with patch("src.api.rag_pipeline.rag_query", side_effect=RuntimeError("token=secret")):
            with self.assertRaises(main.HTTPException) as caught:
                asyncio.run(main.query_endpoint(
                    main.QueryRequest(query="Explain anticipatory bail conditions")
                ))
        main.retriever = None
        self.assertEqual(caught.exception.status_code, 500)
        self.assertNotIn("secret", caught.exception.detail)

    def test_postgres_driver_unavailable_is_optional(self):
        with (
            patch.dict(os.environ, {"DATABASE_URL": "postgresql://test"}),
            patch.object(main, "psycopg2", None),
        ):
            main.log_to_postgres("q", "a", 0.5, 0.5, 1, 0, False, "General")
            result = asyncio.run(main.stats())
        self.assertEqual(result, {"message": "PostgreSQL unavailable"})

    def test_lightweight_evaluation_runner_with_synthetic_fixture(self):
        import tempfile
        from pathlib import Path
        fake_query = Mock(return_value={
            "answer": "Market value is determined on notification date.",
            "scores": {"relevance": 0.8, "faithfulness": 0.9},
            "metadata": {"self_healed": False, "detected_intent": "Land&Property"},
            "sources": [],
        })
        with tempfile.TemporaryDirectory() as directory, patch("src.evaluation.eval_runner.time.sleep"):
            output = Path(directory) / "results.json"
            summary = run_evaluation(
                "tests/fixtures/eval_tiny.json", str(output),
                retriever=object(), query_fn=fake_query
            )
            self.assertEqual(summary["total_questions"], 1)
            self.assertEqual(summary["avg_relevance"], 0.8)
            self.assertGreater(summary["avg_answer_overlap"], 0.1)
            self.assertTrue(check_gate(str(output)))

    def test_existing_evaluation_overlap_method(self):
        self.assertGreater(compute_token_overlap("market value on notification date", "market value determined on notification date"), 0)

    def test_score_gate_uses_existing_thresholds(self):
        import json
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", encoding="utf-8", delete=False) as file:
            json.dump({"summary": {
                "avg_relevance": 0.6,
                "avg_faithfulness": 0.7,
                "avg_answer_overlap": 0.2,
                "execution_pass": True,
            }}, file)
            path = file.name
        try:
            self.assertTrue(check_gate(path))
        finally:
            os.unlink(path)

    def test_score_gate_fails_clearly_when_metric_is_unavailable(self):
        import json
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", encoding="utf-8", delete=False) as file:
            json.dump({"summary": {
                "avg_relevance": 0.8,
                "avg_faithfulness": None,
                "avg_answer_overlap": 0.2,
                "execution_pass": True,
            }}, file)
            path = file.name
        try:
            self.assertFalse(check_gate(path))
        finally:
            os.unlink(path)

    def test_stats_handles_unavailable_postgres(self):
        with patch.dict(os.environ, {"DATABASE_URL": "postgresql://unavailable"}), patch("src.api.main.psycopg2.connect", side_effect=OSError("offline")):
            result = asyncio.run(main.stats())
        self.assertEqual(result["message"], "PostgreSQL unavailable")

    def test_postgres_logging_skips_unconfigured_database(self):
        with patch.dict(os.environ, {}, clear=True), patch("src.api.main.psycopg2.connect") as connect:
            main.log_to_postgres("q", "a", 0.5, 0.5, 1, 0, False, "General")
        connect.assert_not_called()

    def test_postgres_logging_survives_unavailable_database(self):
        with patch.dict(os.environ, {"DATABASE_URL": "postgresql://unavailable"}), patch("src.api.main.psycopg2.connect", side_effect=OSError("offline")):
            main.log_to_postgres("q", "a", 0.5, 0.5, 1, 0, False, "General")


if __name__ == "__main__":
    unittest.main()
