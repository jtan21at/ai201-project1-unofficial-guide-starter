from __future__ import annotations

import argparse
import html
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import chromadb
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer


SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


@dataclass
class Chunk:
    chunk_id: str
    source: str
    chunk_index: int
    text: str


def iter_document_paths(documents_dir: Path) -> Iterable[Path]:
    for path in sorted(documents_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path


def load_document_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="ignore")

    if suffix == ".pdf":
        try:
            import pdfplumber  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "PDF support requires pdfplumber. Install it with `pip install pdfplumber`."
            ) from exc

        text_parts: List[str] = []
        with pdfplumber.open(str(path)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
        return "\n\n".join(text_parts)

    raise ValueError(f"Unsupported file type: {path}")


def clean_text(raw_text: str) -> str:
    text = html.unescape(raw_text)
    text = re.sub(r"<script[\\s\\S]*?</script>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<style[\\s\\S]*?</style>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)

    noisy_patterns = [
        r"\\bcookie(s)?\\b",
        r"\\bprivacy policy\\b",
        r"\\bterms of service\\b",
        r"\\bread more\\b",
        r"\\bshare\\b",
        r"\\bsubscribe\\b",
    ]

    cleaned_lines: List[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        lower = stripped.lower()
        if any(re.search(pattern, lower) for pattern in noisy_patterns):
            continue
        cleaned_lines.append(stripped)

    text = "\n".join(cleaned_lines)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 600, overlap: int = 120) -> List[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    if len(text) <= chunk_size:
        return [text] if text else []

    chunks: List[str] = []
    start = 0
    total = len(text)

    while start < total:
        max_end = min(total, start + chunk_size)
        end = max_end

        if max_end < total:
            search_start = start + int(chunk_size * 0.6)
            sentence_break = text.rfind(". ", search_start, max_end)
            paragraph_break = text.rfind("\n\n", search_start, max_end)
            chosen = max(sentence_break, paragraph_break)
            if chosen > start:
                end = chosen + (2 if chosen == sentence_break else 0)

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= total:
            break

        start = max(end - overlap, start + 1)

    return chunks


def build_chunks(documents_dir: Path, chunk_size: int, overlap: int) -> List[Chunk]:
    all_chunks: List[Chunk] = []

    for path in iter_document_paths(documents_dir):
        raw = load_document_text(path)
        cleaned = clean_text(raw)
        text_chunks = chunk_text(cleaned, chunk_size=chunk_size, overlap=overlap)

        rel_source = str(path.relative_to(documents_dir.parent))
        for idx, chunk in enumerate(text_chunks):
            all_chunks.append(
                Chunk(
                    chunk_id=f"{rel_source}::chunk-{idx}",
                    source=rel_source,
                    chunk_index=idx,
                    text=chunk,
                )
            )

    return all_chunks


def get_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    return SentenceTransformer(model_name)


def get_collection(persist_dir: Path, collection_name: str):
    client = chromadb.PersistentClient(path=str(persist_dir))
    return client, client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})


def index_documents(
    documents_dir: Path,
    persist_dir: Path,
    collection_name: str,
    chunk_size: int,
    overlap: int,
    embedding_model_name: str,
    reindex: bool,
) -> None:
    chunks = build_chunks(documents_dir=documents_dir, chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        raise RuntimeError(f"No chunks built from {documents_dir}. Add documents before indexing.")

    model = get_embedding_model(embedding_model_name)
    embeddings = model.encode([c.text for c in chunks], normalize_embeddings=True).tolist()

    client = chromadb.PersistentClient(path=str(persist_dir))
    if reindex:
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass

    collection = client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})
    collection.upsert(
        ids=[c.chunk_id for c in chunks],
        documents=[c.text for c in chunks],
        metadatas=[{"source": c.source, "chunk_index": c.chunk_index} for c in chunks],
        embeddings=embeddings,
    )

    print(f"Indexed {len(chunks)} chunks into '{collection_name}' at {persist_dir}")
    print("Sample chunks:")
    for sample in chunks[:5]:
        print("-" * 80)
        print(f"{sample.chunk_id}\n{sample.text[:300]}{'...' if len(sample.text) > 300 else ''}")


def retrieve(
    query: str,
    persist_dir: Path,
    collection_name: str,
    embedding_model_name: str,
    top_k: int,
):
    _, collection = get_collection(persist_dir, collection_name)
    if collection.count() == 0:
        raise RuntimeError("Vector store is empty. Run the index command first.")

    model = get_embedding_model(embedding_model_name)
    query_embedding = model.encode([query], normalize_embeddings=True).tolist()[0]

    return collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )


def build_context(retrieval_result: dict) -> str:
    documents = retrieval_result.get("documents", [[]])[0]
    metadatas = retrieval_result.get("metadatas", [[]])[0]

    blocks = []
    for i, (doc, meta) in enumerate(zip(documents, metadatas), start=1):
        source = meta.get("source", "unknown")
        chunk_index = meta.get("chunk_index", "?")
        blocks.append(f"[{i}] Source: {source} (chunk {chunk_index})\n{doc}")

    return "\n\n".join(blocks)


def ask_question(
    question: str,
    persist_dir: Path,
    collection_name: str,
    embedding_model_name: str,
    llm_model: str,
    top_k: int,
) -> None:
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "your_key_here":
        raise RuntimeError("Set GROQ_API_KEY in .env before asking questions.")

    retrieval_result = retrieve(
        query=question,
        persist_dir=persist_dir,
        collection_name=collection_name,
        embedding_model_name=embedding_model_name,
        top_k=top_k,
    )

    context = build_context(retrieval_result)
    if not context.strip():
        print("No relevant context retrieved.")
        return

    system_prompt = (
        "You are a retrieval-grounded assistant for student knowledge. "
        "Answer ONLY using the provided context excerpts. "
        "If the answer is not in the context, say: 'I don't have enough evidence in the retrieved documents.' "
        "Always include citations using [source:chunk] style based on the listed excerpts."
    )

    user_prompt = (
        f"Question: {question}\n\n"
        f"Context excerpts:\n{context}\n\n"
        "Return:\n"
        "1) A concise grounded answer\n"
        "2) A Sources section listing the supporting excerpts"
    )

    client = Groq(api_key=api_key)
    completion = client.chat.completions.create(
        model=llm_model,
        temperature=0.1,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    answer = completion.choices[0].message.content
    print("\n=== Answer ===\n")
    print(answer)

    print("\n=== Retrieved Chunks ===")
    docs = retrieval_result.get("documents", [[]])[0]
    metas = retrieval_result.get("metadatas", [[]])[0]
    dists = retrieval_result.get("distances", [[]])[0]
    for doc, meta, dist in zip(docs, metas, dists):
        print("-" * 80)
        print(f"Source: {meta.get('source')} (chunk {meta.get('chunk_index')}), distance={dist:.4f}")
        print(doc[:300] + ("..." if len(doc) > 300 else ""))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unofficial Guide RAG CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="Ingest, clean, chunk, and index documents")
    index_parser.add_argument("--documents-dir", type=Path, default=Path("documents"))
    index_parser.add_argument("--persist-dir", type=Path, default=Path("chroma_db"))
    index_parser.add_argument("--collection", type=str, default="unofficial-guide")
    index_parser.add_argument("--chunk-size", type=int, default=600)
    index_parser.add_argument("--overlap", type=int, default=120)
    index_parser.add_argument("--embedding-model", type=str, default="all-MiniLM-L6-v2")
    index_parser.add_argument("--reindex", action="store_true")

    ask_parser = subparsers.add_parser("ask", help="Retrieve context and generate grounded answer")
    ask_parser.add_argument("question", type=str)
    ask_parser.add_argument("--persist-dir", type=Path, default=Path("chroma_db"))
    ask_parser.add_argument("--collection", type=str, default="unofficial-guide")
    ask_parser.add_argument("--embedding-model", type=str, default="all-MiniLM-L6-v2")
    ask_parser.add_argument("--llm-model", type=str, default="llama-3.3-70b-versatile")
    ask_parser.add_argument("--top-k", type=int, default=4)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "index":
        index_documents(
            documents_dir=args.documents_dir,
            persist_dir=args.persist_dir,
            collection_name=args.collection,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            embedding_model_name=args.embedding_model,
            reindex=args.reindex,
        )
    elif args.command == "ask":
        ask_question(
            question=args.question,
            persist_dir=args.persist_dir,
            collection_name=args.collection,
            embedding_model_name=args.embedding_model,
            llm_model=args.llm_model,
            top_k=args.top_k,
        )


if __name__ == "__main__":
    main()
