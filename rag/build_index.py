from __future__ import annotations

import argparse
from pathlib import Path

from rag_core import build_index, get_settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a fully local vector index from PDF files.")
    parser.add_argument("--data-dir", type=str, default=None, help="Folder containing PDF files.")
    parser.add_argument("--index-dir", type=str, default=None, help="Folder where the local index will be saved.")
    args = parser.parse_args()

    settings = get_settings()
    if args.data_dir:
        settings.data_dir = Path(args.data_dir)
    if args.index_dir:
        settings.index_dir = Path(args.index_dir)

    pdfs = list(settings.data_dir.glob("*.pdf")) if settings.data_dir.exists() else []
    print(f"Found {len(pdfs)} PDF files in {settings.data_dir}")
    print(f"Embedding model: {settings.embedding_model}")
    print(f"Index folder: {settings.index_dir}")
    manifest = build_index(settings)
    print("\nIndex built successfully.")
    print(f"Documents: {manifest['num_documents']}")
    print(f"Chunks: {manifest['num_chunks']}")
    print(f"Saved to: {settings.index_dir.resolve()}")


if __name__ == "__main__":
    main()
