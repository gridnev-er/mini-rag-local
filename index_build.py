from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import List

from rag.config import load_config
from rag.io_loaders import load_documents
from rag.chunker import Chunk, chunk_documents
from rag.embeddings import EmbeddingEncoder
from rag.index_store import IndexMeta, build_faiss_index, save_index


def _save_chunks_jsonl(chunks: List[Chunk], path: Path) -> None:
    """
    Сохраняем список чанков в простой JSONL-файл:
    одна строка = один чанк в виде словаря.
    Path приводим к строке.
    """
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ch in chunks:
            obj = asdict(ch)
            obj["path"] = str(ch.path)  # Path -> str для JSON
            json_line = json.dumps(obj, ensure_ascii=False)
            f.write(json_line + "\n")


def build_index() -> None:
    """
    Главная функция индексации:

    1. Загружаем конфиг
    2. Читаем документы из data_dir
    3. Режем их на чанки
    4. Считаем эмбеддинги
    5. Строим FAISS-индекс
    6. Сохраняем индекс + чанки в index_dir
    """
    cfg = load_config()
    data_dir: Path = cfg["paths"]["data_dir"]
    index_dir: Path = cfg["paths"]["index_dir"]

    print(f"[BUILD] data_dir = {data_dir}")
    print(f"[BUILD] index_dir = {index_dir}")

    # 1) загрузка документов
    documents = load_documents(data_dir)

    if not documents:
        print("[BUILD] Документов не найдено, индексация прервана.")
        return

    # 2) чанкинг
    chunk_size = cfg["chunking"]["chunk_size"]
    overlap = cfg["chunking"]["overlap"]

    chunks = chunk_documents(
        documents=documents,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    if not chunks:
        print("[BUILD] Чанков не получилось, индексация прервана.")
        return

    # 3) эмбеддинги
    encoder = EmbeddingEncoder()
    embeddings = encoder.embed_chunks(chunks)  # (N, dim)

    # 4) FAISS-индекс
    index = build_faiss_index(embeddings)
    row_ids = [ch.id for ch in chunks]
    meta = IndexMeta.from_embedding_meta(encoder.meta, num_vectors=len(row_ids))

    # 5) сохранение индекса и чанков
    save_index(index_dir, index, row_ids, meta)

    chunks_path = index_dir / "chunks.jsonl"
    _save_chunks_jsonl(chunks, chunks_path)
    print(f"[BUILD] Сохранены чанки в {chunks_path} (count={len(chunks)}).")

    print("[BUILD] Индексация завершена успешно.")


if __name__ == "__main__":
    build_index()
