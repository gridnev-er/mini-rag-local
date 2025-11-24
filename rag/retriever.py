from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Sequence

import numpy as np

from .chunker import Chunk
from .embeddings import EmbeddingEncoder
from .index_store import LoadedIndex


Status = Literal["ok", "insufficient"]


@dataclass
class RetrievedChunk:
    chunk_id: int # глобальный id чанка
    path: str # путь к файлу (относительно data_dir)
    text: str # текст чанка
    snippet: str # короткий превью-текст
    index_in_doc: int # порядковый номер чанка внутри документа
    score: float # мера близости (чем больше, тем "похожее")


@dataclass
class RetrievalResult:
    status: Status
    query: str # исходный текст запроса
    chunks: List[RetrievedChunk] # список найденных чанков (может быть пустым, если insufficient)


class Retriever:
    def __init__(
        self,
        encoder: EmbeddingEncoder,
        loaded_index: LoadedIndex,
        chunks_by_id: Dict[int, Chunk],
        *,
        min_score: float = 0.25,
    ) -> None:
        self.encoder = encoder
        self.index = loaded_index.index
        self.row_ids = loaded_index.row_ids
        self.index_meta = loaded_index.meta
        self.chunks_by_id = chunks_by_id
        self.min_score = float(min_score)


    def _embed_query(self, query: str) -> np.ndarray:

        # Считает эмбеддинг запроса и приводит к форме (1, dim), подходящей для FAISS.search().
        vec = self.encoder.embed_query(query)  # (dim,)
        return vec.reshape(1, -1)

    def _deduplicate_chunks(
        self,
        chunk_ids: Sequence[int],
        texts: Sequence[str],
    ) -> List[int]:
        
        # Убираем дубликаты по точному совпадению текста. Возвращаем список chunk_id без повторов в исходном порядке.
        seen_texts = set()
        unique_ids: List[int] = []

        for cid, txt in zip(chunk_ids, texts):
            if txt in seen_texts:
                continue
            seen_texts.add(txt)
            unique_ids.append(cid)

        return unique_ids

    def search(
        self,
        query_text: str,
        top_k: int,
    ) -> RetrievalResult:
        
        # Основной метод: по тексту запроса возвращает список релевантных чанков.
        query_text = (query_text or "").strip()
        if not query_text:
            raise ValueError("Текст запроса не должен быть пустым.")

        if top_k <= 0:
            raise ValueError("top_k должен быть >= 1.")

        # 1) эмбеддинг запроса
        query_vec = self._embed_query(query_text)  # (1, dim)

        # 2) поиск в индексе
        # FAISS вернёт:
        # - scores: (1, top_k) — меры похожести
        # - indices: (1, top_k) — номера строк в индексе
        scores, indices = self.index.search(query_vec, k=top_k)

        raw_scores = scores[0]
        raw_indices = indices[0]

        # Когда индекс пустой или FAISS не нашёл ничего — индексы могут быть -1.
        # Защитимся от этого: считаем валидными только >= 0.
        valid_pairs = [
            (int(row_idx), float(score))
            for row_idx, score in zip(raw_indices, raw_scores)
            if row_idx >= 0
        ]

        if not valid_pairs:
            return RetrievalResult(status="insufficient", query=query_text, chunks=[])

        # 3) маппинг row_id -> chunk_id
        candidate_chunk_ids: List[int] = []
        candidate_scores: List[float] = []

        for row_idx, score in valid_pairs:
            if row_idx >= len(self.row_ids):
                continue  # на всякий случай
            chunk_id = int(self.row_ids[row_idx])
            candidate_chunk_ids.append(chunk_id)
            candidate_scores.append(score)

        if not candidate_chunk_ids:
            return RetrievalResult(status="insufficient", query=query_text, chunks=[])

        # 4) фильтрация по min_score
        filtered_ids: List[int] = []
        filtered_scores: List[float] = []

        for cid, score in zip(candidate_chunk_ids, candidate_scores):
            if score < self.min_score:
                continue
            filtered_ids.append(cid)
            filtered_scores.append(score)

        if not filtered_ids:
            # Есть индекс, но всё слишком "далеко" от запроса
            return RetrievalResult(status="insufficient", query=query_text, chunks=[])

        # 5) убираем дубликаты по тексту
        texts_for_dedupe: List[str] = []
        for cid in filtered_ids:
            chunk = self.chunks_by_id.get(cid)
            if chunk is None:
                texts_for_dedupe.append("")  # для консистентности длины
            else:
                texts_for_dedupe.append(chunk.text)

        unique_ids = self._deduplicate_chunks(filtered_ids, texts_for_dedupe)

        # 6) собираем RetrievedChunk (в исходной сортировке по score)
        retrieved: List[RetrievedChunk] = []
        for cid in unique_ids:
            chunk = self.chunks_by_id.get(cid)
            if chunk is None:
                continue

            # score ищем по исходному списку filtered_ids
            try:
                idx = filtered_ids.index(cid)
                score = filtered_scores[idx]
            except ValueError:
                score = 0.0

            retrieved.append(
                RetrievedChunk(
                    chunk_id=cid,
                    path=str(chunk.path),
                    text=chunk.text,
                    snippet=chunk.snippet,
                    index_in_doc=chunk.index_in_doc,
                    score=score,
                )
            )

        if not retrieved:
            return RetrievalResult(status="insufficient", query=query_text, chunks=[])

        return RetrievalResult(status="ok", query=query_text, chunks=retrieved)


if __name__ == "__main__":
    # Небольшой самотест без файлов:
    # создаём парочку чанков в памяти, считаем по ним эмбеддинги,
    # строим индекс и проверяем, что ретривер находит их по запросу.
    from pathlib import Path

    import numpy as np

    from .embeddings import EmbeddingEncoder
    from .index_store import IndexMeta, LoadedIndex, build_faiss_index

    print("[RETR] Самотест: локальный поиск по двум чанкам.")

    # 1) фиктивные чанки
    chunks: List[Chunk] = [
        Chunk(
            id=0,
            path=Path("data/doc1.txt"),
            text="RAG — это подход, который комбинирует поиск по документам и LLM.",
            index_in_doc=0,
            snippet="RAG — это подход...",
        ),
        Chunk(
            id=1,
            path=Path("data/doc2.txt"),
            text="FAISS — это библиотека для быстрого векторного поиска.",
            index_in_doc=0,
            snippet="FAISS — это библиотека...",
        ),
    ]

    # 2) считаем эмбеддинги
    encoder = EmbeddingEncoder()
    embs = encoder.embed_chunks(chunks)  # (2, dim)

    # 3) строим индекс и маппинг row_id -> chunk_id
    index = build_faiss_index(embs)
    row_ids = np.array([c.id for c in chunks], dtype="int64")
    meta = IndexMeta.from_embedding_meta(encoder.meta, num_vectors=len(row_ids))
    loaded_index = LoadedIndex(index=index, row_ids=row_ids, meta=meta)

    # 4) словарь chunk_id -> Chunk
    chunks_by_id = {c.id: c for c in chunks}

    # 5) создаём ретривер и делаем запрос
    retriever = Retriever(encoder, loaded_index, chunks_by_id, min_score=0.2)

    query = "Что такое FAISS и для чего он нужен?"
    result = retriever.search(query, top_k=2)

    print("[RETR] Статус:", result.status)
    for rc in result.chunks:
        print("---")
        print("chunk_id:", rc.chunk_id)
        print("path:", rc.path)
        print("score:", rc.score)
        print("snippet:", rc.snippet)