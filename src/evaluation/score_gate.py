import json
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
    passed = True

    logger.info("=== SCORE GATE CHECK ===")
    for metric, threshold in THRESHOLDS.items():
        actual = summary.get(metric, 0.0)
        status = "✓ PASS" if actual >= threshold else "✗ FAIL"
        logger.info(
            f"{metric}: {actual} "
            f"(threshold: {threshold}) — {status}"
        )
        if actual < threshold:
            passed = False

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