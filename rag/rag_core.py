from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


@dataclass
class Settings:
    data_dir: Path
    index_dir: Path
    embedding_model: str
    chunk_size_words: int
    chunk_overlap_words: int
    top_k: int


def get_settings() -> Settings:
    load_dotenv()
    return Settings(
        data_dir=Path(os.getenv("DATA_DIR", "rag/data")),
        index_dir=Path(os.getenv("INDEX_DIR", "rag/index")),
        embedding_model=os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small"),
        chunk_size_words=int(os.getenv("CHUNK_SIZE_WORDS", "420")),
        chunk_overlap_words=int(os.getenv("CHUNK_OVERLAP_WORDS", "80")),
        top_k=int(os.getenv("TOP_K", "8")),
    )


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def find_pdfs(data_dir: Path) -> list[Path]:
    if not data_dir.exists():
        raise FileNotFoundError(f"PDF folder not found: {data_dir}")
    return sorted(data_dir.glob("*.pdf"))


def extract_pdf_pages(pdf_path: Path) -> list[dict[str, Any]]:
    reader = PdfReader(str(pdf_path))
    pages: list[dict[str, Any]] = []
    for page_index, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text() or "")
        if text:
            pages.append(
                {
                    "source": pdf_path.name,
                    "path": str(pdf_path),
                    "page": page_index,
                    "text": text,
                }
            )
    return pages


def chunk_words(words: list[str], size: int, overlap: int) -> Iterable[tuple[int, int, list[str]]]:
    if size <= 0:
        raise ValueError("chunk size must be positive")
    if overlap >= size:
        raise ValueError("chunk overlap must be smaller than chunk size")

    start = 0
    while start < len(words):
        end = min(start + size, len(words))
        yield start, end, words[start:end]
        if end == len(words):
            break
        start = max(0, end - overlap)


def build_chunks(
    pdf_paths: list[Path],
    chunk_size_words: int,
    chunk_overlap_words: int,
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    chunk_id = 0

    for pdf_path in pdf_paths:
        for page in extract_pdf_pages(pdf_path):
            words = page["text"].split()
            for start, end, part in chunk_words(words, chunk_size_words, chunk_overlap_words):
                chunk_text = " ".join(part).strip()
                if len(chunk_text) < 80:
                    continue
                chunks.append(
                    {
                        "id": chunk_id,
                        "source": page["source"],
                        "path": page["path"],
                        "page": page["page"],
                        "word_start": start,
                        "word_end": end,
                        "text": chunk_text,
                    }
                )
                chunk_id += 1

    return chunks


def _uses_e5(model_name: str) -> bool:
    return "e5" in model_name.lower()


def _format_passages_for_embedding(texts: list[str], model_name: str) -> list[str]:
    if _uses_e5(model_name):
        return [f"passage: {text}" for text in texts]
    return texts


def _format_query_for_embedding(query: str, model_name: str) -> str:
    if _uses_e5(model_name):
        return f"query: {query}"
    return query


def load_embedding_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def build_index(settings: Settings) -> dict[str, Any]:
    pdfs = find_pdfs(settings.data_dir)
    if not pdfs:
        raise FileNotFoundError(f"No PDF files found in {settings.data_dir}")

    chunks = build_chunks(pdfs, settings.chunk_size_words, settings.chunk_overlap_words)
    if not chunks:
        raise RuntimeError("No text chunks were extracted from the PDFs.")

    settings.index_dir.mkdir(parents=True, exist_ok=True)
    model = load_embedding_model(settings.embedding_model)

    texts = [chunk["text"] for chunk in chunks]
    formatted_texts = _format_passages_for_embedding(texts, settings.embedding_model)
    embeddings = model.encode(
        formatted_texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).astype("float32")

    np.save(settings.index_dir / "embeddings.npy", embeddings)
    with (settings.index_dir / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    manifest = {
        "embedding_model": settings.embedding_model,
        "chunk_size_words": settings.chunk_size_words,
        "chunk_overlap_words": settings.chunk_overlap_words,
        "documents": [pdf.name for pdf in pdfs],
        "num_documents": len(pdfs),
        "num_chunks": len(chunks),
    }
    with (settings.index_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    return manifest


class LocalRAG:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.manifest = self._load_manifest()
        self.embeddings = self._load_embeddings()
        self.chunks = self._load_chunks()
        self.model = load_embedding_model(self.manifest["embedding_model"])

    def _load_manifest(self) -> dict[str, Any]:
        path = self.settings.index_dir / "manifest.json"
        if not path.exists():
            raise FileNotFoundError(
                f"Index not found in {self.settings.index_dir}. "
                "Run: python rag/build_index.py --data-dir rag/data --index-dir rag/index"
            )
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _load_embeddings(self) -> np.ndarray:
        path = self.settings.index_dir / "embeddings.npy"
        if not path.exists():
            raise FileNotFoundError(f"Missing embeddings file: {path}")
        return np.load(path)

    def _load_chunks(self) -> list[dict[str, Any]]:
        path = self.settings.index_dir / "chunks.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Missing chunks file: {path}")
        chunks = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                chunks.append(json.loads(line))
        return chunks

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        top_k = top_k or self.settings.top_k
        model_name = self.manifest["embedding_model"]
        formatted_query = _format_query_for_embedding(query, model_name)
        query_embedding = self.model.encode(
            [formatted_query],
            normalize_embeddings=True,
        ).astype("float32")[0]
        scores = self.embeddings @ query_embedding
        top_idx = np.argsort(scores)[::-1][:top_k]

        results: list[dict[str, Any]] = []
        for idx in top_idx:
            chunk = dict(self.chunks[int(idx)])
            chunk["score"] = float(scores[int(idx)])
            results.append(chunk)
        return results
