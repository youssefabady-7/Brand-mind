"""Document Ingestion, Structural Chunking, and Metadata Tagging.

Enforces embedding model token budget (<= 250 tokens) to prevent tail truncation
by all-MiniLM-L6-v2 (Risk R1).
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger("brandmind.chunker")

RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")
OUTPUT_CHUNKS_FILE = PROCESSED_DATA_DIR / "chunks.jsonl"


def infer_doc_type(file_name: str) -> str:
    """Infers the document type category from filename stems."""
    lower = file_name.lower()
    if "shipping" in lower or "delivery" in lower:
        return "shipping_policy"
    if "return" in lower or "exchange" in lower or "refund" in lower:
        return "return_policy"
    if "routine" in lower or "usage" in lower or "care" in lower:
        return "usage_guide"
    if "faq" in lower or "safety" in lower:
        return "faq"
    if "escalation" in lower or "rules" in lower:
        return "internal_escalation"
    if "story" in lower or "values" in lower or "vibe" in lower:
        return "brand_story"
    return "general_knowledge"


def approximate_token_count(text: str) -> int:
    """Estimates word-piece token count for sentence-transformers models (~1.3 tokens per word)."""
    words = text.split()
    return int(len(words) * 1.3)


def split_section_into_subchunks(
    header_prefix: str,
    body_text: str,
    max_tokens: int = 220,
    overlap_tokens: int = 30,
) -> List[str]:
    """Splits a large section into overlapping chunks bounded by max_tokens."""
    paragraphs = [p.strip() for p in body_text.split("\n\n") if p.strip()]
    if not paragraphs:
        combined = f"{header_prefix}\n{body_text}".strip()
        return [combined] if combined else []

    chunks: List[str] = []
    current_paragraphs: List[str] = []
    current_tokens = approximate_token_count(header_prefix)

    for p in paragraphs:
        p_tokens = approximate_token_count(p)

        # If a single paragraph is too large, split by sentences
        if p_tokens > max_tokens:
            sentences = re.split(r"(?<=[.!?])\s+", p)
            for s in sentences:
                s_tokens = approximate_token_count(s)
                if current_tokens + s_tokens > max_tokens and current_paragraphs:
                    chunk_text = f"{header_prefix}\n" + "\n".join(current_paragraphs)
                    chunks.append(chunk_text.strip())
                    # Keep last paragraph or tail for overlap
                    current_paragraphs = [current_paragraphs[-1]] if current_paragraphs else []
                    current_tokens = approximate_token_count(header_prefix) + (
                        approximate_token_count(current_paragraphs[0]) if current_paragraphs else 0
                    )
                current_paragraphs.append(s)
                current_tokens += s_tokens
        else:
            if current_tokens + p_tokens > max_tokens and current_paragraphs:
                chunk_text = f"{header_prefix}\n" + "\n".join(current_paragraphs)
                chunks.append(chunk_text.strip())
                current_paragraphs = [current_paragraphs[-1]] if current_paragraphs else []
                current_tokens = approximate_token_count(header_prefix) + (
                    approximate_token_count(current_paragraphs[0]) if current_paragraphs else 0
                )
            current_paragraphs.append(p)
            current_tokens += p_tokens

    if current_paragraphs:
        chunk_text = f"{header_prefix}\n" + "\n".join(current_paragraphs)
        chunks.append(chunk_text.strip())

    return chunks


def parse_markdown_document(
    file_path: Path,
    brand_id: str,
    max_tokens: int = 220,
) -> List[Dict[str, Any]]:
    """Parses a markdown document into structured chunks with rich metadata.

    Args:
        file_path: Path to the raw markdown file.
        brand_id: Target brand identifier ('lumora' or 'urbankicks').
        max_tokens: Conservative token threshold to avoid embedding truncation.

    Returns:
        List of chunk dictionaries with metadata.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    doc_name = file_path.name
    doc_stem = file_path.stem
    doc_type = infer_doc_type(doc_name)
    now_iso = datetime.now(timezone.utc).isoformat()

    # Extract Document H1 Title
    h1_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
    doc_title = h1_match.group(1).strip() if h1_match else doc_stem.replace("_", " ").title()

    # Split document by H2 / H3 sections
    section_pattern = re.compile(r"^(#{2,3}\s+.+)$", re.MULTILINE)
    splits = section_pattern.split(content)

    chunks: List[Dict[str, Any]] = []
    chunk_idx = 1

    # If document has preamble before first H2
    preamble = splits[0].strip()
    if preamble and len(preamble) > 30:
        header_prefix = f"[{brand_id.title()} | {doc_title} | Overview]"
        subchunks = split_section_into_subchunks(header_prefix, preamble, max_tokens=max_tokens)
        for sc in subchunks:
            chunk_id = f"{brand_id}_{doc_stem}_{chunk_idx:03d}"
            chunks.append({
                "chunk_id": chunk_id,
                "brand_id": brand_id,
                "doc_name": doc_name,
                "doc_type": doc_type,
                "section": "Overview",
                "text": sc,
                "token_count": approximate_token_count(sc),
                "last_updated": now_iso,
            })
            chunk_idx += 1

    # Process each section
    for i in range(1, len(splits), 2):
        heading_line = splits[i].strip()
        section_title = re.sub(r"^#{2,3}\s+", "", heading_line).strip()
        body_text = splits[i + 1].strip() if i + 1 < len(splits) else ""

        if not body_text:
            continue

        header_prefix = f"[{brand_id.title()} | {doc_title} | {section_title}]"
        subchunks = split_section_into_subchunks(header_prefix, body_text, max_tokens=max_tokens)

        for sc in subchunks:
            chunk_id = f"{brand_id}_{doc_stem}_{chunk_idx:03d}"
            chunks.append({
                "chunk_id": chunk_id,
                "brand_id": brand_id,
                "doc_name": doc_name,
                "doc_type": doc_type,
                "section": section_title,
                "text": sc,
                "token_count": approximate_token_count(sc),
                "last_updated": now_iso,
            })
            chunk_idx += 1

    return chunks


def process_all_raw_documents(
    raw_dir: Path = RAW_DATA_DIR,
    output_file: Path = OUTPUT_CHUNKS_FILE,
    max_tokens: int = 220,
) -> List[Dict[str, Any]]:
    """Scans all raw brand documents and outputs unified processed chunks JSONL."""
    all_chunks: List[Dict[str, Any]] = []

    for brand_folder in raw_dir.iterdir():
        if brand_folder.is_dir():
            brand_id = brand_folder.name.lower()
            for doc_file in brand_folder.glob("*.md"):
                doc_chunks = parse_markdown_document(doc_file, brand_id, max_tokens=max_tokens)
                all_chunks.extend(doc_chunks)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    logger.info(f"Successfully processed {len(all_chunks)} chunks into {output_file}")
    return all_chunks


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    chunks = process_all_raw_documents()
    print(f"Generated {len(chunks)} chunks across brands.")
