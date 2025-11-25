from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from .chunker import Chunk


@dataclass
class EmbeddingMeta:
    model_name: str
    dim: int


class EmbeddingEncoder:

    # Цель: один раз загрузить модель эмбеддингов, уметь посчитать эмбеддинги для списка текстов/чанков, держать в себе мету (имя модели, размерность).
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str | None = None,
    ) -> None:
        """
        model_name: имя модели из sentence-transformers
        device: можно указать "cpu" или "cuda", по умолчанию авто-выбор
        """
        self.model_name = model_name
        self._model = SentenceTransformer(model_name, device=device)
        self.dim: int = int(self._model.get_sentence_embedding_dimension())

    def _to_numpy_float32(self, vectors) -> np.ndarray:
        """
        Приводим вывод модели к numpy float32.
        На выходе всегда (N, dim).
        """
        arr = np.asarray(vectors)
        if arr.dtype != np.float32:
            arr = arr.astype("float32")
        # если пришёл вектор (dim,), превращаем в (1, dim)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr

    @property
    def meta(self) -> EmbeddingMeta:

        # Удобный доступ к метаданным
        return EmbeddingMeta(model_name=self.model_name, dim=self.dim)

    def embed_texts(
        self,
        texts: Sequence[str], # список текстов
        batch_size: int = 32, # размер батча для ускорения
        normalize: bool = True, # если True — L2-нормализация
    ) -> np.ndarray:

        if not texts:
            # возвращаем "пустую" матрицу нужной ширины
            return np.zeros((0, self.dim), dtype="float32")

        vectors = self._model.encode(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
        )
        return self._to_numpy_float32(vectors)

    def embed_chunks(
        self,
        chunks: Sequence[Chunk],
        batch_size: int = 32,
        normalize: bool = True,
    ) -> np.ndarray:
        
        # Считает эмбеддинги для списка чанков.
        texts: List[str] = [c.text for c in chunks]
        return self.embed_texts(texts, batch_size=batch_size, normalize=normalize)

    def embed_query(
        self,
        query: str,
        normalize: bool = True,
    ) -> np.ndarray:

        # Возвращает вектор формы (dim,), чтобы потом можно было сделать query_vec.reshape(1, -1) и скормить в FAISS.
        
        if not query:
            raise ValueError("Текст запроса не должен быть пустым.")

        vectors = self.embed_texts([query], batch_size=1, normalize=normalize)
        # embed_texts вернул (1, dim) → берём единственную строку
        return vectors[0]