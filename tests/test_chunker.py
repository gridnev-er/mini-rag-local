from pathlib import Path

import pytest

from rag.io_loaders import Document
from rag.chunker import chunk_documents


def make_doc(text: str, name: str = "doc.txt") -> Document:
    return Document(
        path=Path(name),
        kind="txt",
        text=text,
        num_chars=len(text),
        file_size=len(text.encode("utf-8")),  # простой вариант размера файла в байтах
    )


def test_chunk_documents_basic():
    text = "abcdefghijklmnopqrstuvwxyz"  # 26 символов
    doc = make_doc(text)

    chunks = chunk_documents([doc], chunk_size=10, overlap=2)

    # Должно получиться несколько чанков
    assert len(chunks) >= 2

    # Первый чанк — index_in_doc = 0
    assert chunks[0].index_in_doc == 0

    # Чанки идут по порядку
    indices = [ch.index_in_doc for ch in chunks]
    assert indices == list(range(len(chunks)))

    # У каждого чанка есть текст и сниппет
    for ch in chunks:
        assert ch.text.strip() != ""
        assert isinstance(ch.snippet, str)
        assert len(ch.snippet) > 0
        assert len(ch.snippet) <= 200  # по реализации _make_snippet


def test_chunk_documents_invalid_params():
    doc = make_doc("hello world")

    # chunk_size <= 0
    with pytest.raises(ValueError):
        chunk_documents([doc], chunk_size=0, overlap=0)

    # overlap < 0
    with pytest.raises(ValueError):
        chunk_documents([doc], chunk_size=10, overlap=-1)

    # overlap >= chunk_size
    with pytest.raises(ValueError):
        chunk_documents([doc], chunk_size=10, overlap=10)
