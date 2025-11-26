from rag.citations import (
    build_citations,
    format_citations_markdown,
    _normalize_snippet,
)
from rag.retriever import RetrievedChunk


def test_normalize_snippet_basic():
    text = "строка 1\nстрока 2\nстрока 3"
    snippet = _normalize_snippet(text, max_len=50)

    # Переводы строк должны превратиться в пробелы
    assert "\n" not in snippet
    assert "строка 1" in snippet
    assert "строка 2" in snippet

    # Если текст длиннее max_len — должна быть многоточие
    long_text = "слово " * 100
    long_snippet = _normalize_snippet(long_text, max_len=50)
    assert long_snippet.endswith("...")


def test_build_citations_dedup_and_snippet_use():
    chunks = [
        RetrievedChunk(
            chunk_id=1,
            path="data/doc1.txt",
            text="Полный текст первого чанка.",
            snippet="Сниппет 1",
            index_in_doc=0,
            score=0.9,
        ),
        # Дубликат по chunk_id — должен быть отброшен
        RetrievedChunk(
            chunk_id=1,
            path="data/doc1.txt",
            text="Другая версия текста, которой быть не должно.",
            snippet="Сниппет 1 (дубликат)",
            index_in_doc=1,
            score=0.8,
        ),
        RetrievedChunk(
            chunk_id=2,
            path="data/doc2.txt",
            text="Полный текст второго чанка.",
            snippet="Сниппет 2",
            index_in_doc=0,
            score=0.7,
        ),
    ]

    citations = build_citations(chunks)

    # Всего должно остаться 2 уникальных цитаты
    assert len(citations) == 2

    # Индексы должны идти с 1
    assert citations[0].index == 1
    assert citations[1].index == 2

    # Используется snippet, а не полный text
    assert "Сниппет 1" in citations[0].snippet
    assert "Сниппет 2" in citations[1].snippet


def test_format_citations_markdown_empty():
    md = format_citations_markdown([])
    assert md.strip() == "Источники:\n—"


def test_format_citations_markdown_with_scores():
    chunks = [
        RetrievedChunk(
            chunk_id=1,
            path="data/doc1.txt",
            text="Полный текст.",
            snippet="Сниппет",
            index_in_doc=0,
            score=0.12345,
        )
    ]
    citations = build_citations(chunks)
    md = format_citations_markdown(citations, show_scores=True)

    assert "Источники:" in md
    assert "data/doc1.txt" in md
    assert "Сниппет" in md
    # score с тремя знаками после запятой
    assert "0.123" in md
