from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from .io_loaders import Document  # наш класс из блока 3


@dataclass
class Chunk:
    id: int   # глобальный идентификатор чанка
    path: Path  # относительный путь к исходному файлу внутри data_dir
    text: str  # текст чанка, с которым будут считаться эмбеддинги / работать LLM
    index_in_doc: int  # порядковый номер чанка внутри документа (0, 1, 2, ...)
    snippet: str  # короткое превью чанка (первые N символов без лишних переносов)


def _make_snippet(text: str, max_len: int = 200) -> str:

    # Делает короткое превью из текста чанка: берём первые max_len символов, заменяем переводы строк на пробелы, тримим по краям
    if not text:
        return ""
    snippet = text[:max_len].replace("\n", " ").strip()
    return snippet


def chunk_documents(
    documents: List[Document],
    chunk_size: int,
    overlap: int,
) -> List[Chunk]:

    # Принимает список Document и параметры чанкинга, возвращает список Chunk.
    if chunk_size <= 0:
        raise ValueError("chunk_size должен быть > 0.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap должен быть в диапазоне 0 ≤ overlap < chunk_size.")

    chunks: List[Chunk] = []
    global_chunk_id = 0  # простой автоинкремент для id

    for doc in documents:
        text = doc.text
        if not text:
            continue

        start = 0
        index_in_doc = 0
        length = len(text)
        step = chunk_size - overlap

        while start < length:
            end = start + chunk_size
            chunk_text = text[start:end]

            # Защита от суперкоротких хвостов, чтобы не плодить много чанков
            if len(chunk_text.strip()) < 5:
                break

            snippet = _make_snippet(chunk_text)

            chunk = Chunk(
                id=global_chunk_id,
                path=doc.path,
                text=chunk_text,
                index_in_doc=index_in_doc,
                snippet=snippet,
            )
            chunks.append(chunk)

            global_chunk_id += 1
            index_in_doc += 1
            start += step

    print(
        f"[CHUNKER] Получено документов: {len(documents)}, "
        f"создано чанков: {len(chunks)}."
    )

    return chunks


if __name__ == "__main__":

    from .config import load_config
    from .io_loaders import load_documents

    cfg = load_config()
    docs = load_documents(cfg["paths"]["data_dir"])

    chunk_size = cfg["chunking"]["chunk_size"]
    overlap = cfg["chunking"]["overlap"]

    chunks = chunk_documents(docs, chunk_size=chunk_size, overlap=overlap)

    print("--- Пример чанка ---")
    if chunks:
        ch = chunks[0]
        print("id:", ch.id)
        print("path:", ch.path)
        print("index_in_doc:", ch.index_in_doc)
        print("snippet:", ch.snippet)
