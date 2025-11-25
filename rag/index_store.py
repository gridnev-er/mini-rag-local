from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import List

import faiss  # faiss-cpu
import numpy as np

from .embeddings import EmbeddingMeta


@dataclass
class IndexMeta:
    embedding_model: str # имя модели эмбеддингов
    dim: int # размерность векторов
    index_type: str # тип FAISS-индекса
    built_at: str # когда индекс был построен (ISO-строка)
    num_vectors: int # сколько векторов добавлено в индекс

    @classmethod
    def from_embedding_meta(cls, emb_meta: EmbeddingMeta, num_vectors: int) -> IndexMeta:
        return cls(
            embedding_model=emb_meta.model_name,
            dim=emb_meta.dim,
            index_type="IndexFlatIP",
            built_at=datetime.utcnow().isoformat() + "Z",
            num_vectors=int(num_vectors),
        )

    @classmethod
    def from_dict(cls, data: dict) -> IndexMeta:
        return cls(
            embedding_model=data["embedding_model"],
            dim=int(data["dim"]),
            index_type=data["index_type"],
            built_at=data["built_at"],
            num_vectors=int(data["num_vectors"]),
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class LoadedIndex:
    index: faiss.Index # FAISS-индекс
    row_ids: np.ndarray # numpy-массив chunk_id для каждой строки индекса
    meta: IndexMeta # метаданные индекса


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:

    # Строит FAISS IndexFlatIP по матрице эмбеддингов.
    if embeddings.ndim != 2:
        raise ValueError(f"Ожидается массив 2D (num_vectors, dim), а не {embeddings.shape}.")

    num_vectors, dim = embeddings.shape
    if num_vectors == 0:
        raise ValueError("Нельзя строить индекс по пустому набору эмбеддингов.")

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)  # добавляем все вектора

    print(f"[INDEX] Построен IndexFlatIP: dim={dim}, vectors={num_vectors}.")
    return index


def save_index(
    index_dir: Path | str,
    index: faiss.Index,
    row_ids: np.ndarray,
    meta: IndexMeta,
) -> None:
    
    # Сохраняет индекс и сопутствующие данные в каталог index_dir.
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    index_path = index_dir / "faiss.index"
    row_ids_path = index_dir / "row_ids.npy"
    meta_path = index_dir / "meta.json"

    # приводим row_ids к int64 (удобный и безопасный тип)
    row_ids = np.asarray(row_ids, dtype="int64")

    faiss.write_index(index, str(index_path))
    np.save(row_ids_path, row_ids)

    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta.to_dict(), f, ensure_ascii=False, indent=2)

    print(
        f"[INDEX] Индекс сохранён в {index_dir} "
        f"(vectors={meta.num_vectors}, dim={meta.dim})."
    )


def load_index(index_dir: Path | str) -> LoadedIndex:
    
    # Загружает индекс, row_ids и метаданные из каталога index_dir.
    index_dir = Path(index_dir)

    index_path = index_dir / "faiss.index"
    row_ids_path = index_dir / "row_ids.npy"
    meta_path = index_dir / "meta.json"

    if not index_path.exists() or not row_ids_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"В каталоге {index_dir} не найден полный набор файлов индекса "
            "(faiss.index, row_ids.npy, meta.json). "
            "Сначала построй индекс с помощью save_index()."
        )

    index = faiss.read_index(str(index_path))
    row_ids = np.load(row_ids_path)

    with meta_path.open("r", encoding="utf-8") as f:
        meta_dict = json.load(f)
    meta = IndexMeta.from_dict(meta_dict)

    # sanity-check
    if index.d != meta.dim:
        raise RuntimeError(
            f"Размерность индекса ({index.d}) не совпадает с meta.dim ({meta.dim})."
        )

    if index.ntotal != len(row_ids) or index.ntotal != meta.num_vectors:
        raise RuntimeError(
            "Несогласованность количества векторов: "
            f"index.ntotal={index.ntotal}, len(row_ids)={len(row_ids)}, "
            f"meta.num_vectors={meta.num_vectors}."
        )

    print(
        f"[INDEX] Индекс загружен из {index_dir} "
        f"(vectors={meta.num_vectors}, dim={meta.dim})."
    )

    return LoadedIndex(index=index, row_ids=row_ids, meta=meta)