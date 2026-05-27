from __future__ import annotations

import argparse
from pathlib import Path

from local_rag import LocalRAG, get_settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask a question to the local RAG chatbot.")
    parser.add_argument("question", type=str, help="Question to ask.")
    parser.add_argument("--top-k", type=int, default=None, help="Number of chunks to retrieve.")
    parser.add_argument("--index-dir", type=str, default=None, help="Folder containing the local index.")
    args = parser.parse_args()

    settings = get_settings()
    if args.index_dir:
        settings.index_dir = Path(args.index_dir)

    rag = LocalRAG(settings)
    result = rag.answer(args.question, top_k=args.top_k)
    print(result["answer"])
    print("\nSources:")
    for source in result["sources"]:
        print(f"- {source['source']} p.{source['page']} score={source['score']:.3f}")


if __name__ == "__main__":
    main()
