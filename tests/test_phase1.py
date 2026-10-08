import asyncio
import os
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
groq = types.ModuleType("groq")
groq.Groq = Mock(return_value=Mock())
sys.modules.setdefault("groq", groq)
os.environ.setdefault("GROQ_API_KEY", "test-key")

from src.api import main, rag_pipeline
from src.chunking.legal_chunker import chunk_all_judgments, parse_args
from src.retrieval.hybrid_retriever import HybridRetriever
from src.evaluation.eval_runner import compute_token_overlap
from src.evaluation.score_gate import check_gate
from src.evaluation.eval_runner import run_evaluation


class PhaseOneTests(unittest.TestCase):
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
        with patch.object(rag_pipeline.groq_client.chat.completions, "create", return_value=response):
            rag_pipeline.rag_query("Explain the applicable legal test", retriever, filter_case_type="Tax")
        self.assertEqual(retriever.retrieve.call_args.kwargs["filter_case_type"], "Tax")

    def test_api_forwards_filter_case_type(self):
        request = main.QueryRequest(query="Explain anticipatory bail conditions", filter_case_type="Criminal")
        main.retriever = object()
        result = {"query": request.query, "answer": "Answer", "sources": [], "scores": {"relevance": 0.8, "faithfulness": 0.8}, "metadata": {"detected_intent": "Criminal", "self_healed": False, "chunks_retrieved": 0, "model": "test"}}
        with patch("src.api.rag_pipeline.rag_query", return_value=result) as query:
            asyncio.run(main.query_endpoint(request))
        self.assertEqual(query.call_args.kwargs["filter_case_type"], "Criminal")
        main.retriever = None

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
            json.dump({"summary": {"avg_relevance": 0.6, "avg_faithfulness": 0.7, "avg_answer_overlap": 0.2}}, file)
            path = file.name
        try:
            self.assertTrue(check_gate(path))
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
