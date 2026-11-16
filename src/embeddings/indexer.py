"""ChromaDB Indexing and Embedding Generation Pipeline.

Handles vectorization of structured markdown chunks into a persistent ChromaDB
collection with cosine similarity space configuration (Risk R4).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sys

# Ensure repository root is on sys.path for direct script execution
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from typing import Any, Dict, List, Optional

logger = logging.getLogger("brandmind.embeddings")

DEFAULT_CHROMA_PATH = Path("data/chroma_db")
CHUNKS_FILE = Path("data/processed/chunks.jsonl")
COLLECTION_NAME = "brandmind_knowledge"
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class VectorStoreIndexer:
    """Manages chunk vectorization and persistence inside local ChromaDB."""

    def __init__(
        self,
        persist_dir: Path = DEFAULT_CHROMA_PATH,
        collection_name: str = COLLECTION_NAME,
        model_name: str = DEFAULT_MODEL_NAME,
    ):
        self.persist_dir = Path(persist_dir)
        self.collection_name = collection_name
        self.model_name = model_name
        self._client = None
        self._model = None
        self._collection = None

    def _init_chroma(self):
        """Lazy initialization of ChromaDB client and collection."""
        if self._client is None:
            import chromadb
            from chromadb.config import Settings

            self.persist_dir.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=str(self.persist_dir),
                settings=Settings(anonymized_telemetry=False),
            )
            # Use cosine distance metric: cosine distance = 1 - cosine similarity
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )

    def _get_embedding_model(self):
        """Lazy loads sentence-transformers embedding model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def index_chunks(
        self,
        chunks_file: Path = CHUNKS_FILE,
        batch_size: int = 32,
    ) -> int:
        """Indexes all chunks from JSONL into ChromaDB collection.

        Args:
            chunks_file: Path to processed chunks JSONL.
            batch_size: Embedding batch size.

        Returns:
            Total count of indexed chunks.
        """
        if not chunks_file.exists():
            raise FileNotFoundError(f"Chunks file not found at {chunks_file}. Run chunker first.")

        self._init_chroma()
        model = self._get_embedding_model()

        chunks: List[Dict[str, Any]] = []
        with open(chunks_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    chunks.append(json.loads(line))

        if not chunks:
            logger.warning("No chunks found to index.")
            return 0

        logger.info(f"Beginning indexing for {len(chunks)} chunks into '{self.collection_name}'...")

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            ids = [c["chunk_id"] for c in batch]
            texts = [c["text"] for c in batch]
            metadatas = [
                {
                    "brand_id": c["brand_id"],
                    "doc_name": c["doc_name"],
                    "doc_type": c["doc_type"],
                    "section": c["section"],
                    "token_count": int(c.get("token_count", 0)),
                    "last_updated": c.get("last_updated", ""),
                }
                for c in batch
            ]

            embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=False)
            embeddings_list = embeddings.tolist()

            self._collection.upsert(
                ids=ids,
                documents=texts,
                embeddings=embeddings_list,
                metadatas=metadatas,
            )

        total_count = self._collection.count()
        logger.info(f"Indexing complete. Collection '{self.collection_name}' now holds {total_count} chunks.")
        return total_count


def index_knowledge_base():
    """Convenience function to run chunking and indexing end-to-end."""
    from src.retrieval.chunker import process_all_raw_documents

    process_all_raw_documents()
    indexer = VectorStoreIndexer()
    return indexer.index_chunks()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    count = index_knowledge_base()
    print(f"Total chunks indexed in ChromaDB: {count}")
