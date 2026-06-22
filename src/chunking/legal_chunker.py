import json
import re
import logging
from pathlib import Path
from dataclasses import dataclass, asdict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s"
)
logger = logging.getLogger(__name__)

# Why these specific sizes:
# - 1000 chars ≈ 250 tokens — fits well within context with room for query
# - 200 char overlap — ensures citations at chunk boundaries aren't lost
# Legal sentences are long — 200 char overlap catches cross-boundary references
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MIN_CHUNK_SIZE = 100  # below this a chunk is noise, not signal


@dataclass
class LegalChunk:
    """
    One unit of retrieval in your RAG system.
    Every field here becomes either a vector or a metadata filter.
    Design this carefully — you can't change it without re-embedding everything.
    """
    # content
    chunk_id: str          # unique ID — used to retrieve from ChromaDB
    text: str              # the actual text that gets embedded
    text_length: int       # for analytics

    # document identity — metadata filters
    judgment_title: str
    year: str
    court_type: str
    court_name_normalized: str
    case_type: str
    petitioner: str
    respondent: str
    doc_url: str

    # position within document
    chunk_index: int       # 0, 1, 2... within this judgment
    total_chunks: int      # how many chunks this judgment produced

    # citation graph inputs
    cites_count: int
    cited_by_count: int

    # section classification — filled later by classifier
    predicted_section: str = "UNKNOWN"
    section_confidence: float = 0.0


def split_into_sentences(text: str) -> list[str]:
    """
    Split legal text into sentences respecting legal patterns.

    Why not just split on '.' ?
    Legal text has:
    - "S.C.R." — abbreviations with dots
    - "Section 138." — numbered items ending with dot
    - "Rs. 50,000" — currency with dot
    - "Hon'ble Mr. Justice" — titles with dots
    Naive split on '.' destroys all of these.
    """
    # pattern: split on period/exclamation/question
    # but NOT when preceded by single capital letter (abbreviation)
    # and NOT when followed by lowercase (mid-sentence)
    sentence_endings = re.compile(
        r'(?<!\b[A-Z])(?<!\b[A-Z][a-z])(?<=[.!?])\s+(?=[A-Z])'
    )
    sentences = sentence_endings.split(text)

    # also split on numbered paragraphs — common in Indian judgments
    # pattern: "\n\n1." or "\n2." at start of line
    result = []
    for sentence in sentences:
        # split on paragraph numbers
        para_splits = re.split(r'\n+(?=\d+\.\s)', sentence)
        result.extend(para_splits)

    return [s.strip() for s in result if s.strip()]


def create_chunks_from_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP
) -> list[str]:
    """
    Create overlapping chunks from legal text.

    Strategy:
    1. Split into sentences first — never cut mid-sentence
    2. Accumulate sentences until chunk_size reached
    3. Start next chunk with last 'overlap' characters of previous chunk
       — this ensures citations at boundaries are captured in both chunks

    Why sentence-boundary chunking beats token chunking for law:
    - A legal argument is one unit of meaning
    - Cutting "the court held that the accused is guilty" at word 4
      gives you "the court held that" — meaningless for retrieval
    - Sentence-boundary chunks preserve complete legal thoughts
    """
    if not text or len(text) < MIN_CHUNK_SIZE:
        return []

    sentences = split_into_sentences(text)
    if not sentences:
        return [text[:chunk_size]] if len(text) >= MIN_CHUNK_SIZE else []

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        # if adding this sentence exceeds chunk_size
        if len(current_chunk) + len(sentence) > chunk_size:
            # save current chunk if it meets minimum size
            if len(current_chunk.strip()) >= MIN_CHUNK_SIZE:
                chunks.append(current_chunk.strip())

            # start new chunk with overlap from end of previous chunk
            if chunks:
                overlap_text = current_chunk[-overlap:] if len(
                    current_chunk) > overlap else current_chunk
                current_chunk = overlap_text + " " + sentence
            else:
                current_chunk = sentence
        else:
            current_chunk += " " + sentence if current_chunk else sentence

    # don't forget the last chunk
    if current_chunk.strip() and len(
            current_chunk.strip()) >= MIN_CHUNK_SIZE:
        chunks.append(current_chunk.strip())

    return chunks


def chunk_judgment(judgment: dict, judgment_idx: int) -> list[LegalChunk]:
    """
    Convert a single processed judgment into a list of LegalChunks.
    This is the core unit of your ingestion pipeline.
    """
    text = judgment.get("cleaned_text", "")
    title = judgment.get("title", "")

    if not text:
        return []

    # guard against monster documents — cap at 500k chars for now
    # a 2.4M char document would produce 2400 chunks — too many
    # in production you'd handle this differently (hierarchical chunking)
    if len(text) > 500_000:
        logger.warning(
            f"Document truncated: {title[:50]} "
            f"({len(text):,} chars → 500,000)"
        )
        text = text[:500_000]

    raw_chunks = create_chunks_from_text(text)

    if not raw_chunks:
        return []

    chunks = []
    for i, chunk_text in enumerate(raw_chunks):
        chunk_id = f"judgment_{judgment_idx:06d}_chunk_{i:04d}"

        chunk = LegalChunk(
            chunk_id=chunk_id,
            text=chunk_text,
            text_length=len(chunk_text),

            judgment_title=title,
            year=judgment.get("year", "Unknown"),
            court_type=judgment.get("court_type", ""),
            court_name_normalized=judgment.get("court_name_normalized", ""),
            case_type=judgment.get("case_type", ""),
            petitioner=judgment.get("petitioner", ""),
            respondent=judgment.get("respondent", ""),
            doc_url=judgment.get("doc_url", ""),

            chunk_index=i,
            total_chunks=len(raw_chunks),

            cites_count=judgment.get("cites_count", 0),
            cited_by_count=judgment.get("cited_by_count", 0),
        )
        chunks.append(chunk)

    return chunks


def chunk_all_judgments(
    judgments_path: str,
    output_path: str,
    max_judgments: int = None
) -> list[dict]:
    """
    Process all judgments into chunks.
    max_judgments — useful during development to test on subset.
    Set to None for full dataset.
    """
    judgments_path = Path(judgments_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(judgments_path) as f:
        judgments = json.load(f)

    if max_judgments:
        judgments = judgments[:max_judgments]
        logger.info(f"Using subset: {max_judgments} judgments")

    all_chunks = []
    total_skipped = 0

    for i, judgment in enumerate(judgments):
        if i % 500 == 0:
            logger.info(
                f"Chunking judgment {i}/{len(judgments)} "
                f"— chunks so far: {len(all_chunks)}"
            )

        chunks = chunk_judgment(judgment, i)

        if not chunks:
            total_skipped += 1
            continue

        all_chunks.extend([asdict(c) for c in chunks])

    # save to disk
    with open(output_path, "w") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    logger.info(f"Total chunks: {len(all_chunks)}")
    logger.info(f"Skipped judgments: {total_skipped}")
    logger.info(f"Avg chunks per judgment: "
                f"{len(all_chunks)/max(len(judgments),1):.1f}")
    logger.info(f"Saved to: {output_path}")

    return all_chunks


def analyze_chunks(chunks: list[dict]):
    """
    Understand your chunks before embedding them.
    Always do this — bad chunks waste embedding API costs.
    """
    lengths = [c['text_length'] for c in chunks]
    case_types = {}
    court_types = {}

    for c in chunks:
        ct = c['case_type']
        court = c['court_type']
        case_types[ct] = case_types.get(ct, 0) + 1
        court_types[court] = court_types.get(court, 0) + 1

    print("\n=== CHUNK ANALYSIS ===")
    print(f"Total chunks: {len(chunks):,}")
    print(f"Min length: {min(lengths):,} chars")
    print(f"Max length: {max(lengths):,} chars")
    print(f"Avg length: {sum(lengths)//len(lengths):,} chars")

    print("\nChunks by case type:")
    for k, v in sorted(case_types.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v:,}")

    print("\nChunks by court type:")
    for k, v in sorted(court_types.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v:,}")

    print("\nSample chunk:")
    print("-" * 50)
    print(chunks[100]['text'][:300])
    print("-" * 50)


if __name__ == "__main__":
    # during development use max_judgments=500 to test fast
    # remove limit for full dataset
    chunks = chunk_all_judgments(
        judgments_path="data/processed/judgments_clean.json",
        output_path="data/processed/chunks.json",
        max_judgments=500
    )
    analyze_chunks(chunks)