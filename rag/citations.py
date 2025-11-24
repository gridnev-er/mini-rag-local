from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

from .retriever import RetrievedChunk


@dataclass
class SourceCitation:
    index: int # порядковый номер
    path: str # путь к файлу
    snippet: str # короткий текст-превью
    score: Optional[float] = None # опционально, мера релевантности 


def _normalize_snippet(text: str, max_len: int = 180) -> str:
    
    # Делает из полного текста чанка короткий сниппет
    if not text:
        return ""

    # заменяем переводы строк на пробелы
    snippet = " ".join(text.split())

    if len(snippet) <= max_len:
        return snippet

    # аккуратно обрезаем по слову
    cut = snippet[:max_len]
    last_space = cut.rfind(" ")
    if last_space > 0:
        cut = cut[:last_space]

    return cut + "..."


def build_citations(
    chunks: Sequence[RetrievedChunk],
    *,
    use_existing_snippet: bool = True,
) -> List[SourceCitation]:
    
    # Превращает список RetrievedChunk в список SourceCitation.
    citations: List[SourceCitation] = []
    seen_chunk_ids = set()

    for idx, ch in enumerate(chunks, start=1):
        if ch.chunk_id in seen_chunk_ids:
            # защитимся от случайных дублей
            continue
        seen_chunk_ids.add(ch.chunk_id)

        if use_existing_snippet and ch.snippet:
            snippet = _normalize_snippet(ch.snippet)
        else:
            snippet = _normalize_snippet(ch.text)

        citations.append(
            SourceCitation(
                index=idx,
                path=ch.path,
                snippet=snippet,
                score=ch.score,
            )
        )

    return citations


def format_citations_markdown(
    citations: Sequence[SourceCitation],
    *,
    show_scores: bool = False,
) -> str:
    
    # Формирует Markdown-блок "Источники" по списку SourceCitation.
    if not citations:
        return "Источники:\n—"

    lines: List[str] = ["Источники:"]
    for src in citations:
        base = f"{src.index}) {src.path} — \"{src.snippet}\""
        if show_scores and src.score is not None:
            base += f" (score: {src.score:.3f})"
        lines.append(base)

    return "\n".join(lines)


if __name__ == "__main__":

    print("[CIT] Самотест: формирование раздела 'Источники'.")

    fake_chunks = [
        RetrievedChunk(
            chunk_id=0,
            path="data/doc1.txt",
            text="RAG — это подход, который сочетает поиск по документам и LLM. "
                 "Он позволяет отвечать на вопросы только по вашим файлам.",
            snippet="RAG — это подход, который сочетает поиск по документам и LLM.",
            index_in_doc=0,
            score=0.92,
        ),
        RetrievedChunk(
            chunk_id=1,
            path="data/doc2.txt",
            text="FAISS используется для быстрого поиска по векторным эмбеддингам. "
                 "Он хорошо подходит для локальных мини-RAG систем.",
            snippet="FAISS используется для быстрого поиска по векторным эмбеддингам.",
            index_in_doc=0,
            score=0.88,
        ),
    ]

    citations = build_citations(fake_chunks)
    md = format_citations_markdown(citations, show_scores=True)

    print("------ MARKDOWN ------")
    print(md)