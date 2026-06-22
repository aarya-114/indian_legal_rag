import os
import json
import logging
from pathlib import Path
from datasets import load_dataset
from huggingface_hub import login
from dotenv import load_dotenv

load_dotenv()
login(token=os.getenv("HF_TOKEN"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s"
)
logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)


def load_rhetorical_roles() -> dict:
    """
    Load the InRhetoricalRoles dataset.
    Returns dict with train/dev/test splits.
    Each example has:
      - data.text: full judgment text
      - annotations: list of labeled spans with start/end/label
      - meta.group: case category (Criminal, Civil etc)
    """
    logger.info("Loading rhetorical roles dataset...")
    dataset = load_dataset(
        "opennyaiorg/InRhetoricalRoles",
        cache_dir="./data/cache/rhetorical_roles"
    )
    logger.info(f"Loaded — train: {len(dataset['train'])} "
                f"dev: {len(dataset['dev'])} "
                f"test: {len(dataset['test'])}")
    return dataset


def load_judgments() -> dict:
    """
    Load the InJudgements_dataset.
    Returns dict with train split.
    Each example has:
      - Titles: case name
      - Court_Name: originating court
      - Cites: number of cases this judgment cites
      - Cited_by: number of cases that cite this judgment
      - Doc_url: Indian Kanoon URL
      - Text: full judgment text
      - Doc_size: character count
      - Case_Type: legal domain (Land&Property, Criminal etc)
      - Court_Type: High_Court, Supreme_Court etc
      - Court_Name_Normalized: cleaned court name
    """
    logger.info("Loading judgments dataset...")
    dataset = load_dataset(
        "opennyaiorg/InJudgements_dataset",
        cache_dir="./data/cache/judgments"
    )
    logger.info(f"Loaded — train: {len(dataset['train'])} judgments")
    return dataset


def extract_judgment_metadata(example: dict) -> dict:
    """
    Extract structured metadata from a single judgment example.
    This becomes one row in your PostgreSQL metadata table.
    """
    return {
        "title": example.get("Titles", ""),
        "court_name": example.get("Court_Name", ""),
        "court_name_normalized": example.get("Court_Name_Normalized", ""),
        "court_type": example.get("Court_Type", ""),
        "case_type": example.get("Case_Type", ""),
        "doc_url": example.get("Doc_url", ""),
        "doc_size": example.get("Doc_size", 0),
        "cites_count": example.get("Cites", 0),
        "cited_by_count": example.get("Cited_by", 0),
        "text_length": len(example.get("Text", "")),
        "text_preview": example.get("Text", "")[:200]
    }


def extract_rhetorical_spans(example: dict) -> list[dict]:
    """
    Extract labeled text spans from a rhetorical roles example.
    Each span becomes one training example for your section classifier.

    Returns list of dicts:
      - text: the span text
      - label: rhetorical role (FACTS, ARGUMENTS, RULING etc)
      - start: character start position
      - end: character end position
      - case_group: Criminal/Civil etc
    """
    spans = []
    case_group = example.get("meta", {}).get("group", "Unknown")
    annotations = example.get("annotations", [])

    if not annotations:
        return spans

    # annotations is a list — take the first annotator's results
    results = annotations[0].get("result", [])

    for item in results:
        value = item.get("value", {})
        text = value.get("text", "").strip()
        labels = value.get("labels", [])
        start = value.get("start", 0)
        end = value.get("end", 0)

        if not text or not labels:
            continue

        spans.append({
            "text": text,
            "label": labels[0],  # take first label
            "start": start,
            "end": end,
            "case_group": case_group
        })

    return spans


def save_raw_data(judgments_dataset, rhetorical_dataset):
    """
    Save processed data to disk as JSON for inspection and reuse.
    This is your raw data layer.
    """
    # save judgment metadata
    metadata_path = RAW_DIR / "judgment_metadata.json"
    metadata = []
    for example in judgments_dataset['train']:
        metadata.append(extract_judgment_metadata(example))

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved {len(metadata)} judgment metadata records to {metadata_path}")

    # save rhetorical spans
    spans_path = RAW_DIR / "rhetorical_spans.json"
    all_spans = []
    for example in rhetorical_dataset['train']:
        spans = extract_rhetorical_spans(example)
        all_spans.extend(spans)

    with open(spans_path, "w") as f:
        json.dump(all_spans, f, indent=2)
    logger.info(f"Saved {len(all_spans)} rhetorical spans to {spans_path}")

    return metadata, all_spans


def analyze_data(metadata: list, spans: list):
    """
    Quick analysis of what we have.
    Always analyze your data before building pipelines.
    """
    print("\n=== JUDGMENT ANALYSIS ===")
    print(f"Total judgments: {len(metadata)}")

    court_types = {}
    case_types = {}
    for m in metadata:
        ct = m['court_type']
        ctype = m['case_type']
        court_types[ct] = court_types.get(ct, 0) + 1
        case_types[ctype] = case_types.get(ctype, 0) + 1

    print("\nCourt type distribution:")
    for k, v in sorted(court_types.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")

    print("\nCase type distribution:")
    for k, v in sorted(case_types.items(), key=lambda x: -x[1])[:10]:
        print(f"  {k}: {v}")

    print("\n=== RHETORICAL ROLES ANALYSIS ===")
    print(f"Total labeled spans: {len(spans)}")

    label_counts = {}
    for span in spans:
        label = span['label']
        label_counts[label] = label_counts.get(label, 0) + 1

    print("\nLabel distribution:")
    for k, v in sorted(label_counts.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")

    avg_len = sum(len(s['text']) for s in spans) / len(spans) if spans else 0
    print(f"\nAverage span length: {avg_len:.0f} characters")


if __name__ == "__main__":
    judgments = load_judgments()
    rhetorical = load_rhetorical_roles()
    metadata, spans = save_raw_data(judgments, rhetorical)
    analyze_data(metadata, spans)