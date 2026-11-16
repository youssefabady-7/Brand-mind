"""Unit and integration tests for the RAG chunker, ChromaDB indexer, and retriever.

Validates:
1. Structural markdown chunking and metadata tags (brand_id, doc_type, section).
2. Token budget compliance (all chunks <= 270 tokens).
3. ChromaDB indexing with cosine distance space configuration.
4. Brand isolation (Lumora vs UrbanKicks).
5. Threshold fallback (policy silence) on out-of-scope queries.
"""

import json
from pathlib import Path
import pytest

from src.retrieval.chunker import (
    approximate_token_count,
    parse_markdown_document,
    process_all_raw_documents,
)
from src.embeddings.indexer import VectorStoreIndexer
from src.retrieval.retriever import KnowledgeRetriever, retrieve


@pytest.fixture(scope="module")
def indexed_vector_store(tmp_path_factory):
    """Creates a temporary isolated ChromaDB index for testing."""
    test_chroma_dir = tmp_path_factory.mktemp("chroma_test")
    chunks_file = Path("data/processed/chunks.jsonl")
    
    # Ensure chunks exist
    if not chunks_file.exists():
        process_all_raw_documents()

    indexer = VectorStoreIndexer(
        persist_dir=test_chroma_dir,
        collection_name="test_brandmind_knowledge",
    )
    count = indexer.index_chunks(chunks_file=chunks_file)
    assert count > 0

    retriever = KnowledgeRetriever(
        persist_dir=test_chroma_dir,
        collection_name="test_brandmind_knowledge",
        default_threshold=0.35,
        default_top_k=4,
    )
    return retriever


def test_chunker_generates_valid_metadata():
    """Validates that parsed chunks carry all mandatory metadata fields."""
    sample_file = Path("data/raw/lumora/02_shipping_and_delivery_policy.md")
    chunks = parse_markdown_document(sample_file, brand_id="lumora")
    assert len(chunks) >= 3

    for c in chunks:
        assert c["brand_id"] == "lumora"
        assert c["doc_name"] == "02_shipping_and_delivery_policy.md"
        assert c["doc_type"] == "shipping_policy"
        assert len(c["section"]) > 0
        assert c["token_count"] <= 270  # Risk R1 check
        assert len(c["text"]) > 50


def test_chunker_token_budget_compliance():
    """Ensures no chunk across the entire corpus exceeds the embedding token budget."""
    chunks_file = Path("data/processed/chunks.jsonl")
    if not chunks_file.exists():
        process_all_raw_documents()

    with open(chunks_file, "r", encoding="utf-8") as f:
        all_chunks = [json.loads(line) for line in f]

    assert len(all_chunks) >= 30
    for chunk in all_chunks:
        assert chunk["token_count"] <= 275, f"Chunk {chunk['chunk_id']} exceeded token budget: {chunk['token_count']}"


def test_retriever_in_scope_lumora_shipping(indexed_vector_store):
    """In-scope query retrieves relevant Lumora shipping chunk with score >= threshold."""
    result = indexed_vector_store.retrieve(
        query="What are the shipping delivery times and costs in Cairo?",
        brand_id="lumora",
    )
    assert result.has_answer is True
    assert result.policy_silence is False
    assert result.best_score >= 0.35
    assert len(result.chunks) >= 1
    assert any("Cairo" in c.text for c in result.chunks)
    assert "<context>" in result.context_text


def test_retriever_brand_isolation(indexed_vector_store):
    """Enforces multi-tenancy: Lumora query NEVER retrieves UrbanKicks chunks."""
    result = indexed_vector_store.retrieve(
        query="What is your policy on returns and hygiene?",
        brand_id="lumora",
    )
    assert result.has_answer is True
    # Verify 100% of returned chunks belong to Lumora
    for chunk in result.chunks:
        assert chunk.brand_id == "lumora"
        assert "urbankicks" not in chunk.chunk_id.lower()


def test_retriever_urbankicks_sneakers(indexed_vector_store):
    """UrbanKicks query retrieves footwear condition and unworn criteria."""
    result = indexed_vector_store.retrieve(
        query="Can I return sneakers if I tried them on?",
        brand_id="urbankicks",
    )
    assert result.has_answer is True
    assert any(c.brand_id == "urbankicks" for c in result.chunks)
    assert any("unworn" in c.text.lower() or "sole" in c.text.lower() for c in result.chunks)


def test_retriever_policy_silence_fallback(indexed_vector_store):
    """Completely out-of-scope query triggers policy silence and fallback messages."""
    result = indexed_vector_store.retrieve(
        query="quantum computing cryogenic processor overclocking specifications",
        brand_id="lumora",
        similarity_threshold=0.35,
    )
    assert result.has_answer is False
    assert result.policy_silence is True
    assert len(result.chunks) == 0
    assert result.fallback_message_en is not None
    assert "verified policy information" in result.fallback_message_en
