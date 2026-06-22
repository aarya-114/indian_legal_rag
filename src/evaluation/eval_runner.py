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
    results_path: str
) -> dict:
    """
    Run full evaluation pipeline.
    For each question in eval dataset:
    1. Run through RAG pipeline
    2. Score relevance and faithfulness
    3. Compare answer to ground truth
    4. Track latency and self-heal rate
    """
    from src.retrieval.hybrid_retriever import HybridRetriever
    from src.api.rag_pipeline import rag_query

    logger.info("Loading retriever for evaluation...")
    retriever = HybridRetriever(
        chunks_path="data/processed/chunks.json"
    )

    dataset = load_eval_dataset(eval_dataset_path)
    logger.info(f"Evaluating {len(dataset)} questions...")

    results = []
    total_relevance = 0.0
    total_faithfulness = 0.0
    total_overlap = 0.0
    total_latency = 0.0
    self_heal_count = 0

    for i, item in enumerate(dataset):
        question = item["question"]
        ground_truth = item["ground_truth"]

        logger.info(f"[{i+1}/{len(dataset)}] {question[:60]}...")

        start = time.time()
        try:
            result = rag_query(question, retriever, top_k=5)
            latency_ms = (time.time() - start) * 1000

            # score answer against ground truth
            overlap_score = compute_token_overlap(
                ground_truth,
                result["answer"]
            )

            relevance = result["scores"]["relevance"]
            faithfulness = result["scores"]["faithfulness"]
            self_healed = result["metadata"]["self_healed"]

            total_relevance += relevance
            total_faithfulness += faithfulness
            total_overlap += overlap_score
            total_latency += latency_ms
            if self_healed:
                self_heal_count += 1

            results.append({
                "question": question,
                "ground_truth": ground_truth,
                "answer": result["answer"],
                "expected_case_type": item["expected_case_type"],
                "detected_intent": result["metadata"]["detected_intent"],
                "scores": {
                    "relevance": relevance,
                    "faithfulness": faithfulness,
                    "answer_overlap": overlap_score
                },
                "self_healed": self_healed,
                "latency_ms": round(latency_ms, 2),
                "sources_count": len(result["sources"]),
                "warning": result.get("warning")
            })

            # be nice to Groq rate limits
            time.sleep(1)

        except Exception as e:
            logger.error(f"Failed on question {i+1}: {e}")
            results.append({
                "question": question,
                "error": str(e),
                "scores": {
                    "relevance": 0.0,
                    "faithfulness": 0.0,
                    "answer_overlap": 0.0
                }
            })

    n = len(dataset)
    summary = {
        "total_questions": n,
        "avg_relevance": round(total_relevance / n, 3),
        "avg_faithfulness": round(total_faithfulness / n, 3),
        "avg_answer_overlap": round(total_overlap / n, 3),
        "avg_latency_ms": round(total_latency / n, 2),
        "self_heal_rate": round(self_heal_count / n, 3),
        "self_heal_count": self_heal_count,
        "pass": True  # determined by score_gate.py
    }

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
    logger.info(f"Results saved to:  {results_path}")

    return summary


if __name__ == "__main__":
    summary = run_evaluation(
        eval_dataset_path="data/eval/eval_dataset.json",
        results_path="data/eval/eval_results.json"
    )