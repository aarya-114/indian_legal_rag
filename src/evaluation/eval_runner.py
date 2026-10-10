import argparse
import json
import time
import logging
import sys
import os
from pathlib import Path

sys.path.append(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s"
)
logger = logging.getLogger(__name__)


def load_eval_dataset(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def compute_token_overlap(text1: str, text2: str) -> float:
    """
    Simple token overlap score between ground truth and answer.
    Not perfect but works without external API.
    Production alternative: use RAGAS with OpenAI or a local NLI model.

    Why token overlap?
    If ground truth says "market value on date of notification"
    and answer says "market value is determined on the date of
    the Section 4 notification" — they share key tokens.
    High overlap = answer covers the ground truth concepts.
    """
    tokens1 = set(text1.lower().split())
    tokens2 = set(text2.lower().split())

    # remove common stop words — they add noise
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "be",
        "been", "being", "have", "has", "had", "do", "does",
        "did", "will", "would", "could", "should", "may",
        "might", "shall", "can", "of", "in", "on", "at",
        "to", "for", "with", "by", "from", "and", "or",
        "but", "if", "that", "this", "it", "its", "not"
    }

    tokens1 = tokens1 - stopwords
    tokens2 = tokens2 - stopwords

    if not tokens1 or not tokens2:
        return 0.0

    overlap = tokens1.intersection(tokens2)
    # F1-style score: harmonic mean of precision and recall
    precision = len(overlap) / len(tokens2)
    recall = len(overlap) / len(tokens1)

    if precision + recall == 0:
        return 0.0

    f1 = 2 * (precision * recall) / (precision + recall)
    return round(f1, 3)


def run_evaluation(
    eval_dataset_path: str,
    results_path: str,
    retriever=None,
    query_fn=None
) -> dict:
    """
    Run full evaluation pipeline.
    For each question in eval dataset:
    1. Run through RAG pipeline
    2. Score relevance and faithfulness
    3. Compare answer to ground truth
    4. Track eligible metrics, execution status, latency, and recovery telemetry
    """
    if retriever is None:
        from src.retrieval.hybrid_retriever import HybridRetriever
        logger.info("Loading retriever for evaluation...")
        retriever = HybridRetriever(chunks_path="data/processed/chunks.json")
    if query_fn is None:
        from src.api.rag_pipeline import rag_query
        query_fn = rag_query

    dataset = load_eval_dataset(eval_dataset_path)
    logger.info(f"Evaluating {len(dataset)} questions...")

    results = []
    metric_sums = {
        "relevance": 0.0,
        "faithfulness": 0.0,
        "answer_overlap": 0.0,
    }
    metric_counts = {name: 0 for name in metric_sums}
    total_latency = 0.0
    self_heal_count = 0
    self_heal_questions = 0
    self_heal_eligible_questions = 0
    self_heal_failed_count = 0
    self_heal_unknown_count = 0
    rewrite_attempt_count = 0
    rewrite_success_count = 0
    rewrite_failed_count = 0
    retry_attempt_count = 0
    retry_success_count = 0
    retry_failed_count = 0

    for i, item in enumerate(dataset):
        question = item.get("question", "")
        ground_truth = item.get("ground_truth")
        logger.info(f"[{i+1}/{len(dataset)}] {question[:60]}...")

        relevance = None
        faithfulness = None
        overlap_score = None
        self_healed = None
        rewrite_attempted = None
        retry_attempted = None
        rewrite_succeeded = None
        rewrite_failed = None
        retry_succeeded = None
        retry_failed = None
        fallback_activated = None
        reference_recovered = None
        initial_chunk_ids = []
        fallback_chunk_ids = []
        fallback_candidate_ids = []
        final_chunk_ids = []
        self_heal_status = "unknown"
        answer_status = "failed"
        result = None
        record = {
            "question": question,
            "ground_truth": ground_truth,
            "expected_case_type": item.get("expected_case_type"),
            "answer": None,
            "scores": {},
            "metric_status": {},
        }
        start = time.time()
        try:
            result = query_fn(question, retriever, top_k=5)
            scores = result.get("scores") or {}
            metadata = result.get("metadata") or {}
            answer = result.get("answer")
            relevance = scores.get("relevance")

            has_answer = isinstance(answer, str) and bool(answer.strip())
            answer_status = "completed" if has_answer else "empty"
            faithfulness = scores.get("faithfulness") if has_answer else None

            if not isinstance(ground_truth, str) or not ground_truth.strip():
                overlap_status = "unavailable: reference answer missing"
            elif not has_answer:
                overlap_status = "unavailable: generated answer missing"
            else:
                overlap_score = compute_token_overlap(ground_truth, answer)
                overlap_status = "available"

            self_healed = metadata.get("self_healed")
            rewrite_attempted = metadata.get("rewrite_attempted", False)
            rewrite_succeeded = metadata.get("rewrite_succeeded", False)
            rewrite_failed = metadata.get("rewrite_failed", False)
            retry_attempted = metadata.get("retry_attempted", False)
            retry_succeeded = metadata.get("retry_succeeded", False)
            retry_failed = metadata.get("retry_failed", False)
            fallback_activated = metadata.get("fallback_activated", False)
            reference_recovered = metadata.get("reference_recovered", False)
            initial_chunk_ids = metadata.get("initial_chunk_ids", [])
            fallback_chunk_ids = metadata.get("fallback_chunk_ids", [])
            fallback_candidate_ids = metadata.get("fallback_candidate_ids", [])
            final_chunk_ids = metadata.get("final_chunk_ids", [])
            if fallback_activated:
                self_heal_status = "recovered" if self_healed is True else "failed"
            elif rewrite_attempted is False:
                self_heal_status = "not_triggered"
            elif rewrite_attempted is True and self_healed is True and retry_attempted is True:
                self_heal_status = "recovered"
            elif rewrite_attempted is True and self_healed is False:
                self_heal_status = "failed"
            record.update({
                "answer": answer,
                "answer_status": answer_status,
                "detected_intent": metadata.get("detected_intent"),
                "scores": {
                    "relevance": relevance,
                    "faithfulness": faithfulness,
                    "answer_overlap": overlap_score,
                },
                "metric_status": {
                    "relevance": (
                        "available" if relevance is not None
                        else "unavailable: retrieval score missing"
                    ),
                    "faithfulness": (
                        "available" if faithfulness is not None
                        else "unavailable: generated answer missing"
                    ),
                    "answer_overlap": overlap_status,
                },
                "self_healed": self_healed,
                "self_heal_status": self_heal_status,
                "rewrite_attempted": rewrite_attempted,
                "rewrite_succeeded": rewrite_succeeded,
                "rewrite_failed": rewrite_failed,
                "retry_attempted": retry_attempted,
                "retry_succeeded": retry_succeeded,
                "retry_failed": retry_failed,
                "fallback_activated": fallback_activated,
                "reference_recovered": reference_recovered,
                "initial_chunk_ids": initial_chunk_ids,
                "fallback_candidate_ids": fallback_candidate_ids,
                "fallback_chunk_ids": fallback_chunk_ids,
                "final_chunk_ids": final_chunk_ids,
                "fallback_error": metadata.get("fallback_error"),
                "sources_count": len(result.get("sources") or []),
                "warning": result.get("warning"),
            })
        except Exception as e:
            logger.error(f"Failed on question {i+1}: {e}")
            cause = getattr(e, "cause", e)
            answer_status = (
                "truncated"
                if getattr(cause, "finish_reason", None) == "length"
                else "failed"
            )
            if relevance is None:
                relevance = getattr(e, "relevance_score", None)
            self_healed = getattr(e, "self_healed", self_healed)
            rewrite_attempted = getattr(e, "rewrite_attempted", rewrite_attempted)
            retry_attempted = getattr(e, "retry_attempted", retry_attempted)
            rewrite_succeeded = getattr(e, "rewrite_succeeded", rewrite_succeeded)
            rewrite_failed = getattr(e, "rewrite_failed", rewrite_failed)
            retry_succeeded = getattr(e, "retry_succeeded", retry_succeeded)
            retry_failed = getattr(e, "retry_failed", retry_failed)
            diagnostics = getattr(e, "retrieval_diagnostics", {}) or {}
            fallback_activated = diagnostics.get("fallback_activated", False)
            reference_recovered = diagnostics.get("reference_recovered", False)
            initial_chunk_ids = diagnostics.get("initial_chunk_ids", [])
            fallback_chunk_ids = diagnostics.get("fallback_chunk_ids", [])
            fallback_candidate_ids = diagnostics.get("fallback_candidate_ids", [])
            final_chunk_ids = diagnostics.get("final_chunk_ids", [])
            if fallback_activated:
                self_heal_status = "recovered" if self_healed is True else "failed"
            elif rewrite_attempted is False:
                self_heal_status = "not_triggered"
            elif rewrite_attempted is True and self_healed is True and retry_attempted is True:
                self_heal_status = "recovered"
            elif rewrite_attempted is True and self_healed is False:
                self_heal_status = "failed"
            if overlap_score is not None:
                overlap_status = "available"
            elif not isinstance(ground_truth, str) or not ground_truth.strip():
                overlap_status = "unavailable: reference answer missing"
            elif answer_status == "truncated":
                overlap_status = "unavailable: generation truncated"
            else:
                overlap_status = "unavailable: generation/evaluation failed"
            record.update({
                "answer": result.get("answer") if isinstance(result, dict) else None,
                "error": str(e),
                "failure_stage": getattr(e, "stage", "evaluation"),
                "scores": {
                    "relevance": relevance,
                    "faithfulness": faithfulness,
                    "answer_overlap": overlap_score,
                },
                "metric_status": {
                    "relevance": (
                        "available" if relevance is not None
                        else "unavailable: retrieval score not returned"
                    ),
                    "faithfulness": (
                        "available" if faithfulness is not None
                        else (
                            "unavailable: generation truncated"
                            if answer_status == "truncated"
                            else "unavailable: generation/evaluation failed"
                        )
                    ),
                    "answer_overlap": overlap_status,
                },
                "answer_status": answer_status,
                "self_healed": self_healed,
                "self_heal_status": self_heal_status,
                "rewrite_attempted": rewrite_attempted,
                "rewrite_succeeded": rewrite_succeeded,
                "rewrite_failed": rewrite_failed,
                "retry_attempted": retry_attempted,
                "retry_succeeded": retry_succeeded,
                "retry_failed": retry_failed,
                "fallback_activated": fallback_activated,
                "reference_recovered": reference_recovered,
                "initial_chunk_ids": initial_chunk_ids,
                "fallback_candidate_ids": fallback_candidate_ids,
                "fallback_chunk_ids": fallback_chunk_ids,
                "final_chunk_ids": final_chunk_ids,
                "fallback_error": diagnostics.get("fallback_error"),
                "sources_count": (
                    len(result.get("sources") or [])
                    if isinstance(result, dict) else None
                ),
            })

        latency_ms = (time.time() - start) * 1000
        total_latency += latency_ms
        record["latency_ms"] = round(latency_ms, 2)
        results.append(record)

        for metric, value in (
            ("relevance", relevance),
            ("faithfulness", faithfulness),
            ("answer_overlap", overlap_score),
        ):
            if value is not None:
                metric_sums[metric] += value
                metric_counts[metric] += 1
        if self_healed is not None:
            self_heal_questions += 1
        if self_heal_status == "recovered":
            self_heal_count += 1
        elif self_heal_status == "failed":
            self_heal_failed_count += 1
        elif self_heal_status == "unknown" and rewrite_attempted is True:
            self_heal_unknown_count += 1
        if rewrite_attempted or fallback_activated:
            self_heal_eligible_questions += 1
        if rewrite_attempted:
            rewrite_attempt_count += 1
        if retry_attempted:
            retry_attempt_count += 1
        if rewrite_succeeded:
            rewrite_success_count += 1
        if rewrite_failed:
            rewrite_failed_count += 1
        if retry_succeeded:
            retry_success_count += 1
        if retry_failed:
            retry_failed_count += 1

        # Keep provider throttling outside measured request latency.
        time.sleep(1)

    n = len(dataset)

    def metric_average(name: str, digits: int) -> float | None:
        count = metric_counts[name]
        return round(metric_sums[name] / count, digits) if count else None

    summary = {
        "total_questions": n,
        "avg_relevance": metric_average("relevance", 3),
        "avg_faithfulness": metric_average("faithfulness", 3),
        "avg_answer_overlap": metric_average("answer_overlap", 3),
        "metric_counts": metric_counts,
        "avg_latency_ms": round(total_latency / n, 2) if n else None,
        "self_heal_rate": (
            round(self_heal_count / self_heal_eligible_questions, 3)
            if self_heal_eligible_questions
            else None
        ),
        "self_heal_count": self_heal_count,
        "self_heal_questions": self_heal_questions,
        "self_heal_eligible_questions": self_heal_eligible_questions,
        "successful_recovery_count": self_heal_count,
        "failed_recovery_count": self_heal_failed_count,
        "unknown_recovery_count": self_heal_unknown_count,
        "rewrite_attempt_count": rewrite_attempt_count,
        "rewrite_success_count": rewrite_success_count,
        "rewrite_failed_count": rewrite_failed_count,
        "retry_attempt_count": retry_attempt_count,
        "retry_success_count": retry_success_count,
        "retry_failed_count": retry_failed_count,
    }
    from src.evaluation.score_gate import meets_thresholds
    summary["quality_thresholds_pass"] = meets_thresholds(summary)
    summary["completed_answer_count"] = sum(
        result["answer_status"] == "completed" for result in results
    )
    summary["empty_answer_count"] = sum(
        result["answer_status"] == "empty" for result in results
    )
    summary["failed_answer_count"] = sum(
        result["answer_status"] in {"failed", "truncated"} for result in results
    )
    summary["execution_pass"] = (
        summary["completed_answer_count"] == n
    )
    summary["pass"] = (
        summary["quality_thresholds_pass"] and summary["execution_pass"]
    )

    output = {
        "summary": summary,
        "results": results
    }

    Path(results_path).parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info("=== EVALUATION SUMMARY ===")
    logger.info(f"Avg relevance:     {summary['avg_relevance']}")
    logger.info(f"Avg faithfulness:  {summary['avg_faithfulness']}")
    logger.info(f"Avg answer overlap:{summary['avg_answer_overlap']}")
    logger.info(f"Avg latency:       {summary['avg_latency_ms']}ms")
    logger.info(f"Self-heal rate:    {summary['self_heal_rate']}")
    logger.info(
        "Self-heal recoveries: "
        f"{summary['successful_recovery_count']}/"
        f"{summary['self_heal_eligible_questions']} eligible"
    )
    logger.info(f"Quality thresholds pass: {summary['quality_thresholds_pass']}")
    logger.info(f"Benchmark execution pass: {summary['execution_pass']}")
    logger.info(f"Overall pass:            {summary['pass']}")
    logger.info(f"Results saved to:  {results_path}")

    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Indian Legal RAG evaluation benchmark."
    )
    parser.add_argument(
        "--dataset",
        default="data/eval/eval_dataset.json",
        help="Evaluation dataset JSON path (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        default="data/eval/eval_results.json",
        help="Evaluation results JSON path (default: %(default)s)",
    )
    return parser


def main(argv: list[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)
    return run_evaluation(
        eval_dataset_path=args.dataset,
        results_path=args.output,
    )


if __name__ == "__main__":
    main()
