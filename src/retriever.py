"""Standalone LangChain hybrid retrieval test for the pre-chunked corpus."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import chromadb
from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = PROJECT_ROOT / "data" / "corpus.jsonl"
CHROMA_PATH = PROJECT_ROOT / "data" / "chroma_store"
COLLECTION_NAME = "asterline_corpus_test"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
REQUIRED_FIELDS = {
    "chunk_id",
    "doc_id",
    "title",
    "category",
    "source_url",
    "published",
    "version",
    "text",
}


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    category: str
    source_url: str
    published: str
    version: str
    text: str

    @classmethod
    def from_record(cls, record: dict[str, Any], line_number: int) -> "Chunk":
        missing = REQUIRED_FIELDS - record.keys()
        if missing:
            fields = ", ".join(sorted(missing))
            raise ValueError(f"{CORPUS_PATH}:{line_number}: missing fields: {fields}")

        values = {field: record[field] for field in REQUIRED_FIELDS}
        if any(
            not isinstance(value, str) or not value.strip() for value in values.values()
        ):
            raise ValueError(
                f"{CORPUS_PATH}:{line_number}: all chunk fields must be non-empty strings"
            )
        return cls(**values)

    @property
    def metadata(self) -> dict[str, str]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "title": self.title,
            "category": self.category,
            "source_url": self.source_url,
            "published": self.published,
            "version": self.version,
        }

    def to_document(self) -> Document:
        # BUG 1 FIX: Prepend title and category for indexing so chunks like pricing:1
        # match queries mentioning "plan" or "pricing". Original self.text stays untouched.
        indexed_content = f"{self.title} ({self.category}): {self.text}"
        return Document(page_content=indexed_content, metadata=self.metadata)


@dataclass(frozen=True)
class SearchResult:
    chunk_id: str
    title: str
    category: str
    score: float | None
    method: Literal["vector", "hybrid"]
    text: str
    metadata: dict[str, str]

    def to_prompt_text(self) -> str:
        """BUG 3 FIX: Exposes chunk_id, title, category, published date, and version to the LLM."""
        return (
            f"[{self.chunk_id}] title={self.title!r} | category={self.category} | "
            f"published={self.metadata['published']} | version={self.metadata['version']}\n"
            f"{self.text}"
        )

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)


def load_chunks(path: Path = CORPUS_PATH) -> list[Chunk]:
    """Load existing chunks without splitting or rewriting their text."""
    if not path.is_file():
        raise FileNotFoundError(f"Corpus file does not exist: {path}")

    chunks: list[Chunk] = []
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8-sig") as corpus:
        for line_number, line in enumerate(corpus, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path}:{line_number}: invalid JSON: {error}"
                ) from error
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")

            chunk = Chunk.from_record(record, line_number)
            if chunk.chunk_id in seen_ids:
                raise ValueError(
                    f"{path}:{line_number}: duplicate chunk_id {chunk.chunk_id!r}"
                )
            seen_ids.add(chunk.chunk_id)
            chunks.append(chunk)

    if not chunks:
        raise ValueError(f"Corpus file contains no chunks: {path}")
    return chunks


class HybridRetriever:
    """LangChain Chroma + BM25 retriever."""

    def __init__(
        self,
        chunks: list[Chunk] | None = None,
        *,
        model_name: str = MODEL_NAME,
        chroma_path: Path = CHROMA_PATH,
    ) -> None:
        if chunks is None:
            chunks = load_chunks()
        if not chunks:
            raise ValueError("At least one chunk is required")

        self.chunks = chunks
        self._by_id = {chunk.chunk_id: chunk for chunk in chunks}
        documents = [chunk.to_document() for chunk in chunks]
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            encode_kwargs={"normalize_embeddings": True},
        )

        client = chromadb.PersistentClient(path=str(chroma_path))
        fingerprint = self._fingerprint(chunks, model_name)
        collection = self._get_collection(client)
        if collection is not None and collection.metadata == {
            "fingerprint": fingerprint,
            "embedding_model": model_name,
        }:
            self.vector_store = Chroma(
                client=client,
                collection_name=COLLECTION_NAME,
                embedding_function=self.embeddings,
            )
        else:
            if collection is not None:
                client.delete_collection(COLLECTION_NAME)
            self.vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                ids=[chunk.chunk_id for chunk in chunks],
                client=client,
                collection_name=COLLECTION_NAME,
                collection_metadata={
                    "fingerprint": fingerprint,
                    "embedding_model": model_name,
                },
            )
        self.vector_retriever = self.vector_store.as_retriever(
            search_kwargs={"k": min(6, len(chunks))}
        )
        self.bm25_retriever = BM25Retriever.from_documents(documents)
        self.bm25_retriever.k = min(6, len(chunks))
        self.retriever = EnsembleRetriever(
            retrievers=[self.vector_retriever, self.bm25_retriever],
            weights=[0.6, 0.4],
        )

    def vector_only(self, query: str, top_k: int = 5) -> list[SearchResult]:
        self._validate_query(query, top_k)
        scored_documents = self.vector_store.similarity_search_with_relevance_scores(
            query, k=min(top_k, len(self.chunks))
        )
        return self._format_documents(
            scored_documents, method="vector", has_scores=True
        )

    def hybrid_search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        self._validate_query(query, top_k)
        # BUG 2 FIX: Dynamically scale candidate pool based on requested top_k
        candidate_k = min(max(top_k * 2, 6), len(self.chunks))
        self.vector_retriever.search_kwargs["k"] = candidate_k
        self.bm25_retriever.k = candidate_k

        documents = self.retriever.invoke(query)[:top_k]
        return self._format_documents(documents, method="hybrid")

    def search(self, query: str, top_k: int = 5, mode: str = "hybrid") -> list[SearchResult]:
        if mode == "vector_only":
            return self.vector_only(query, top_k=top_k)
        return self.hybrid_search(query, top_k=top_k)

    def format_chunk(self, result: SearchResult) -> str:
        return result.to_prompt_text()

    @staticmethod
    def _fingerprint(chunks: list[Chunk], model_name: str) -> str:
        payload = {
            "schema_version": "v2_indexed_title_category",
            "model_name": model_name,
            "chunks": [chunk.__dict__ for chunk in chunks],
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
        ).hexdigest()

    @staticmethod
    def _get_collection(
        client: chromadb.ClientAPI,
    ) -> Any | None:
        try:
            return client.get_collection(COLLECTION_NAME)
        except Exception:
            return None

    def _format_documents(
        self,
        documents: list[Document] | list[tuple[Document, float]],
        *,
        method: Literal["vector", "hybrid"],
        has_scores: bool = False,
    ) -> list[SearchResult]:
        results: list[SearchResult] = []
        seen_ids: set[str] = set()
        for item in documents:
            document, score = item if has_scores else (item, None)
            chunk_id = document.metadata["chunk_id"]
            if chunk_id in seen_ids:
                continue
            seen_ids.add(chunk_id)
            chunk = self._by_id[chunk_id]
            results.append(
                SearchResult(
                    chunk_id=chunk.chunk_id,
                    title=chunk.title,
                    category=chunk.category,
                    score=round(float(score), 6) if score is not None else None,
                    method=method,
                    text=chunk.text,
                    metadata=chunk.metadata,
                )
            )
        return results

    @staticmethod
    def _validate_query(query: str, top_k: int) -> None:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer")


LocalKnowledgeBase = HybridRetriever


def main() -> None:
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks from {CORPUS_PATH}")
    retriever = HybridRetriever(chunks)

    for query in [
        "What is the monthly price and retention period for the Team plan?",
        "What is the weekly on-call stipend?",
    ]:
        print(f"\n{'=' * 70}\nQuery: {query}\n{'=' * 70}")
        for result in retriever.hybrid_search(query, top_k=5):
            print(f"\n{result.to_prompt_text()}")


if __name__ == "__main__":
    main()
