import json
import math
import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s"
)
logger = logging.getLogger(__name__)

# thresholds — if any score drops below these, block the PR
THRESHOLDS = {
    "avg_relevance": 0.55,
    "avg_faithfulness": 0.60,
    "avg_answer_overlap": 0.10
}


def meets_thresholds(summary: dict) -> bool:
    """Return whether every required summary metric meets its threshold."""
    for metric, threshold in THRESHOLDS.items():
        actual = summary.get(metric)
        if (
            isinstance(actual, bool)
            or not isinstance(actual, (int, float))
            or not math.isfinite(actual)
            or actual < threshold
        ):
            return False
    return True


def execution_passed(data: dict) -> bool:
    """Require completed, non-empty answers when per-question results exist."""
    summary = data.get("summary") or {}
    declared_status = summary.get("execution_pass")
    results = data.get("results")

    if isinstance(results, list):
        records_complete = all(
            isinstance(result, dict)
            and isinstance(result.get("answer"), str)
            and bool(result["answer"].strip())
            and result.get("answer_status", "completed") == "completed"
            for result in results
        )
        return records_complete and declared_status is not False

    # Summary-only input must explicitly state that execution completed.
    return declared_status is True


def check_gate(results_path: str) -> bool:
    """
    Read eval results and check against thresholds.
    Returns True if all pass, False if any fail.
    Exit code 1 = GitHub Actions marks PR as failed.
    Exit code 0 = PR can be merged.
    """
    with open(results_path) as f:
        data = json.load(f)

    summary = data["summary"]
    thresholds_passed = meets_thresholds(summary)
    run_passed = execution_passed(data)
    passed = thresholds_passed and run_passed

    logger.info("=== SCORE GATE CHECK ===")
    for metric, threshold in THRESHOLDS.items():
        actual = summary.get(metric)
        if not isinstance(actual, (int, float)):
            logger.error(f"{metric}: unavailable or invalid — cannot meet threshold")
            continue
        status = "✓ PASS" if actual >= threshold else "✗ FAIL"
        logger.info(
            f"{metric}: {actual} "
            f"(threshold: {threshold}) — {status}"
        )
    logger.info(
        "Benchmark execution: "
        f"{'complete' if run_passed else 'incomplete'}"
    )
    if passed:
        logger.info("=== ALL CHECKS PASSED — PR approved ===")
    else:
        logger.error("=== SCORE GATE FAILED — PR blocked ===")

    return passed


if __name__ == "__main__":
    results_path = sys.argv[1] if len(sys.argv) > 1 \
        else "data/eval/eval_results.json"

    passed = check_gate(results_path)
    sys.exit(0 if passed else 1)
