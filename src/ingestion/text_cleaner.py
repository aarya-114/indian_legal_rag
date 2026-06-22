import re
import json
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s"
)
logger = logging.getLogger(__name__)


def clean_legal_text(text: str) -> str:
    """
    Clean raw Indian legal judgment text.

    Why each step exists:
    - Legal PDFs converted to text have artifacts: extra spaces,
      page numbers, headers repeating on every page, form feeds.
    - These artifacts destroy chunk quality — a chunk that starts
      with "Page 47" and ends mid-sentence is useless for retrieval.
    - Clean text = better chunks = better retrieval = better answers.
    """
    if not text or not isinstance(text, str):
        return ""

    # remove form feed characters from PDF conversion
    text = text.replace("\f", "\n")

    # normalize unicode quotes and dashes
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2013", "-").replace("\u2014", "-")

    # remove repeated whitespace within lines
    text = re.sub(r"[ \t]+", " ", text)

    # remove page numbers — patterns like "- 47 -" or "Page 47"
    text = re.sub(r"-\s*\d+\s*-", "", text)
    text = re.sub(r"Page\s+\d+\s+of\s+\d+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"Page\s+\d+", "", text, flags=re.IGNORECASE)

    # remove repeated headers (lines that appear 3+ times identically)
    lines = text.split("\n")
    line_counts = {}
    for line in lines:
        stripped = line.strip()
        if stripped:
            line_counts[stripped] = line_counts.get(stripped, 0) + 1

    # filter out lines that repeat more than 3 times — likely headers/footers
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped or line_counts.get(stripped, 0) <= 3:
            cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)

    # collapse 3+ consecutive newlines into 2
    # preserves paragraph breaks without excessive whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)

    # strip leading/trailing whitespace
    text = text.strip()

    return text


def extract_year_from_title(title: str) -> str:
    """
    Extract year from judgment title.
    Titles follow pattern: 'Party A vs Party B on DD Month, YYYY'
    """
    match = re.search(r"\b(19|20)\d{2}\b", title)
    return match.group(0) if match else "Unknown"


def extract_parties_from_title(title: str) -> dict:
    """
    Extract petitioner and respondent from title.
    Pattern: 'Petitioner vs Respondent on date'
    """
    # split on ' vs ' or ' v. ' or ' versus '
    parts = re.split(r"\s+vs\.?\s+|\s+versus\s+|\s+v\.\s+",
                     title, flags=re.IGNORECASE)

    if len(parts) >= 2:
        petitioner = parts[0].strip()
        # respondent may have 'on date' at end — remove it
        respondent = re.sub(r"\s+on\s+\d+.*$", "", parts[1],
                            flags=re.IGNORECASE).strip()
        return {"petitioner": petitioner, "respondent": respondent}

    return {"petitioner": title, "respondent": "Unknown"}


def process_judgment(example: dict) -> dict:
    """
    Process a single judgment into clean, structured format.
    This is the core unit of your data pipeline.

    Input: raw judgment dict from HuggingFace dataset
    Output: cleaned, enriched judgment dict ready for chunking
    """
    title = example.get("Titles", "")
    raw_text = example.get("Text", "")

    cleaned_text = clean_legal_text(raw_text)
    year = extract_year_from_title(title)
    parties = extract_parties_from_title(title)

    return {
        # identity
        "title": title,
        "year": year,
        "petitioner": parties["petitioner"],
        "respondent": parties["respondent"],

        # court info
        "court_name": example.get("Court_Name", ""),
        "court_name_normalized": example.get("Court_Name_Normalized", ""),
        "court_type": example.get("Court_Type", ""),

        # classification
        "case_type": example.get("Case_Type", ""),

        # citation graph inputs
        "cites_count": example.get("Cites", 0),
        "cited_by_count": example.get("Cited_by", 0),
        "doc_url": example.get("Doc_url", ""),

        # text
        "raw_text_length": len(raw_text),
        "cleaned_text": cleaned_text,
        "cleaned_text_length": len(cleaned_text),

        # quality flag — skip very short documents
        "is_valid": len(cleaned_text) > 500
    }


def process_all_judgments(dataset, output_path: str):
    """
    Process entire judgments dataset and save to disk.
    Skips invalid documents and logs progress every 1000 records.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    processed = []
    skipped = 0

    for i, example in enumerate(dataset['train']):
        if i % 1000 == 0:
            logger.info(f"Processing judgment {i}/{len(dataset['train'])}")

        result = process_judgment(example)

        if not result["is_valid"]:
            skipped += 1
            continue

        # drop raw text from processed output — save space
        # cleaned_text is what we use downstream
        processed.append(result)

    with open(output_path, "w") as f:
        json.dump(processed, f, indent=2, ensure_ascii=False)

    logger.info(f"Processed: {len(processed)} judgments")
    logger.info(f"Skipped: {skipped} invalid documents")
    logger.info(f"Saved to: {output_path}")

    return processed


def process_rhetorical_spans(rhetorical_dataset, output_path: str):
    """
    Clean and structure rhetorical role spans for classifier training.
    Removes NONE labels and very short spans.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # labels we actually want to classify
    # dropping NONE — ambiguous, hurts classifier
    valid_labels = {
        "FAC", "ANALYSIS", "PREAMBLE", "PRE_RELIED",
        "ARG_PETITIONER", "RPC", "RLC", "ARG_RESPONDENT",
        "RATIO", "STA", "ISSUE", "PRE_NOT_RELIED"
    }

    cleaned_spans = []
    skipped = 0

    for example in rhetorical_dataset['train']:
        annotations = example.get("annotations", [])
        case_group = example.get("meta", {}).get("group", "Unknown")
        full_text = example.get("data", {}).get("text", "")

        if not annotations:
            continue

        results = annotations[0].get("result", [])

        for item in results:
            value = item.get("value", {})
            text = value.get("text", "").strip()
            labels = value.get("labels", [])

            if not text or not labels:
                continue

            label = labels[0]

            # skip NONE and very short spans (< 20 chars — likely noise)
            if label not in valid_labels or len(text) < 20:
                skipped += 1
                continue

            cleaned_spans.append({
                "text": clean_legal_text(text),
                "label": label,
                "case_group": case_group,
                "text_length": len(text)
            })

    with open(output_path, "w") as f:
        json.dump(cleaned_spans, f, indent=2, ensure_ascii=False)

    logger.info(f"Clean spans: {len(cleaned_spans)}")
    logger.info(f"Skipped spans: {skipped}")
    logger.info(f"Saved to: {output_path}")

    return cleaned_spans


if __name__ == "__main__":
    from datasets import load_dataset
    from huggingface_hub import login
    import os
    from dotenv import load_dotenv

    load_dotenv()
    login(token=os.getenv("HF_TOKEN"))

    judgments = load_dataset(
        "opennyaiorg/InJudgements_dataset",
        cache_dir="./data/cache/judgments"
    )
    rhetorical = load_dataset(
        "opennyaiorg/InRhetoricalRoles",
        cache_dir="./data/cache/rhetorical_roles"
    )

    process_all_judgments(
        judgments,
        "data/processed/judgments_clean.json"
    )
    process_rhetorical_spans(
        rhetorical,
        "data/processed/rhetorical_spans_clean.json"
    )