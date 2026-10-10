import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from src.evaluation.eval_runner import run_evaluation
from src.evaluation.score_gate import check_gate, meets_thresholds


class QueryFailure(RuntimeError):
    stage = "Answer generation"
    relevance_score = 0.8
    self_healed = False
    rewrite_attempted = False
    retry_attempted = False
    cause = RuntimeError("truncated")
    cause.finish_reason = "length"


class EvaluationAuditTests(unittest.TestCase):
    def run_fixture(self, questions, query_fn, sleep_mock=None):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        dataset_path = Path(directory.name) / "dataset.json"
        results_path = Path(directory.name) / "results.json"
        dataset_path.write_text(json.dumps(questions), encoding="utf-8")
        with patch("src.evaluation.eval_runner.time.sleep", sleep_mock or Mock()):
            summary = run_evaluation(
                str(dataset_path), str(results_path),
                retriever=object(), query_fn=query_fn,
            )
        results = json.loads(results_path.read_text(encoding="utf-8"))
        return summary, results, str(results_path)

    def test_summary_pass_matches_score_gate_thresholds(self):
        questions = [
            {"question": "answered", "ground_truth": "the answer"},
            {"question": "truncated", "ground_truth": "the answer"},
        ]

        def query(question, retriever, top_k):
            if question == "truncated":
                raise QueryFailure(
                    "LLM answer generation response truncated: finish_reason=length"
                )
            return {
                "answer": "the answer",
                "scores": {"relevance": 0.9, "faithfulness": 0.9},
                "metadata": {"self_healed": False},
                "sources": [],
            }

        summary, results, path = self.run_fixture(questions, query)

        self.assertTrue(summary["quality_thresholds_pass"])
        self.assertFalse(summary["execution_pass"])
        self.assertFalse(summary["pass"])
        self.assertEqual(summary["failed_answer_count"], 1)
        self.assertEqual(summary["metric_counts"], {
            "relevance": 2, "faithfulness": 1, "answer_overlap": 1,
        })
        self.assertIsNone(results["results"][1]["scores"]["faithfulness"])
        self.assertEqual(results["results"][1]["answer_status"], "truncated")
        self.assertEqual(results["results"][1]["failure_stage"], "Answer generation")
        self.assertIn("finish_reason=length", results["results"][1]["error"])
        self.assertFalse(check_gate(path))
        gate_process = subprocess.run(
            [sys.executable, "src/evaluation/score_gate.py", path],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(gate_process.returncode, 1, gate_process.stdout + gate_process.stderr)

    def test_complete_benchmark_passes_runner_and_gate(self):
        summary, _, path = self.run_fixture(
            [{"question": "complete", "ground_truth": "a useful answer"}],
            lambda question, retriever, top_k: {
                "answer": "a useful answer",
                "scores": {"relevance": 0.8, "faithfulness": 0.8},
                "metadata": {"self_healed": False},
                "sources": [],
            },
        )
        self.assertTrue(summary["execution_pass"])
        self.assertTrue(summary["quality_thresholds_pass"])
        self.assertTrue(summary["pass"])
        self.assertTrue(check_gate(path))
        gate_process = subprocess.run(
            [sys.executable, "src/evaluation/score_gate.py", path],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(gate_process.returncode, 0, gate_process.stdout + gate_process.stderr)

    def test_empty_generation_is_excluded_from_answer_quality_averages(self):
        summary, results, _ = self.run_fixture(
            [{"question": "empty", "ground_truth": "expected answer"}],
            lambda question, retriever, top_k: {
                "answer": "",
                "scores": {"relevance": 0.8, "faithfulness": 0.0},
                "metadata": {"self_healed": False},
                "sources": [],
            },
        )

        record = results["results"][0]
        self.assertIsNone(record["scores"]["faithfulness"])
        self.assertIsNone(record["scores"]["answer_overlap"])
        self.assertEqual(record["answer_status"], "empty")
        self.assertEqual(
            record["metric_status"]["faithfulness"],
            "unavailable: generated answer missing",
        )
        self.assertEqual(summary["metric_counts"], {
            "relevance": 1, "faithfulness": 0, "answer_overlap": 0,
        })
        self.assertFalse(summary["pass"])
        self.assertFalse(summary["execution_pass"])
        self.assertEqual(summary["self_heal_eligible_questions"], 0)
        self.assertIsNone(summary["self_heal_rate"])

    def test_score_gate_rejects_below_threshold_or_invalid_metrics(self):
        below_threshold = {
            "avg_relevance": 0.9,
            "avg_faithfulness": 0.59,
            "avg_answer_overlap": 0.9,
        }
        self.assertFalse(meets_thresholds(below_threshold))
        invalid = {
            "avg_relevance": 0.9,
            "avg_faithfulness": "0.9",
            "avg_answer_overlap": 0.9,
        }
        self.assertFalse(meets_thresholds(invalid))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.json"
            path.write_text(json.dumps({
                "summary": {**below_threshold, "execution_pass": True},
            }), encoding="utf-8")
            self.assertFalse(check_gate(str(path)))

            path.write_text(json.dumps({
                "summary": {
                    "avg_relevance": 0.9,
                    "avg_faithfulness": 0.9,
                    "avg_answer_overlap": 0.9,
                },
            }), encoding="utf-8")
            self.assertFalse(check_gate(str(path)))

    def test_latency_excludes_rate_limit_sleep(self):
        clock = [10.0]

        def query(question, retriever, top_k):
            clock[0] += 0.25
            return {
                "answer": "answer",
                "scores": {"relevance": 0.8, "faithfulness": 0.8},
                "metadata": {"self_healed": False},
                "sources": [],
            }

        sleep = Mock(side_effect=lambda delay: clock.__setitem__(0, clock[0] + delay))
        with patch("src.evaluation.eval_runner.time.time", side_effect=lambda: clock[0]):
            summary, results, _ = self.run_fixture(
                [{"question": "timed", "ground_truth": "answer"}], query,
                sleep_mock=sleep,
            )

        self.assertEqual(results["results"][0]["latency_ms"], 250.0)
        self.assertEqual(summary["avg_latency_ms"], 250.0)
        sleep.assert_called_once_with(1)


if __name__ == "__main__":
    unittest.main()
