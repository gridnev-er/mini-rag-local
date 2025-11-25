from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple, Any

import gradio as gr

from rag.config import load_config
from rag.embeddings import EmbeddingEncoder
from rag.index_store import load_index
from rag.chunker import Chunk
from rag.retriever import Retriever
from rag.generator import GenerationConfig, AnswerGenerator
from rag.citations import format_citations_markdown


def _load_chunks_jsonl(chunks_path: Path) -> Dict[int, Chunk]:
    """
    Читает chunks.jsonl и возвращает словарь {chunk_id -> Chunk}.

    Ожидается, что файл был записан через asdict(Chunk) в index_build.py.
    """
    if not chunks_path.exists():
        raise FileNotFoundError(
            f"Не найден файл чанков {chunks_path}. "
            "Сначала запусти index_build.py, чтобы собрать индекс."
        )

    chunks_by_id: Dict[int, Chunk] = {}

    with chunks_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Некорректная JSON-строка в {chunks_path} на линии {line_num}: {e}"
                ) from e

            # path в JSON — строка, а в Chunk — Path
            if "path" in data:
                data["path"] = Path(data["path"])

            try:
                chunk = Chunk(**data)
            except TypeError as e:
                raise TypeError(
                    f"Не удалось создать Chunk из данных в {chunks_path} "
                    f"на линии {line_num}: {data}"
                ) from e

            chunks_by_id[chunk.id] = chunk

    if not chunks_by_id:
        raise RuntimeError(f"Файл {chunks_path} не содержит ни одного чанка.")

    print(f"[APP] Загружено чанков: {len(chunks_by_id)} из {chunks_path}.")
    return chunks_by_id


def init_pipeline(config_path: str = "config.yaml") -> Tuple[dict, Retriever, AnswerGenerator]:
    """
    Собирает весь RAG-пайплайн:

    config.yaml  -> пути, top_k, модель
    index/       -> FAISS-индекс + chunks.jsonl
    EmbeddingEncoder -> модель эмбеддингов
    Retriever    -> поиск релевантных чанков
    AnswerGenerator -> вызов LLM и формирование ответа

    Возвращает:
        cfg, retriever, generator
    """
    # 1) Конфиг
    cfg: dict[str, Any] = load_config(config_path)

    index_dir = Path(cfg["paths"]["index_dir"]).resolve()
    if not index_dir.exists():
        raise FileNotFoundError(
            f"Каталог индекса {index_dir} не найден. "
            "Сначала запусти index_build.py, чтобы собрать индекс."
        )

    # 2) FAISS-индекс
    loaded_index = load_index(index_dir)

    # 3) Чанки
    chunks_path = index_dir / "chunks.jsonl"
    chunks_by_id = _load_chunks_jsonl(chunks_path)

    # 4) Модель эмбеддингов (должна совпадать с той, что использовалась при индексации)
    encoder = EmbeddingEncoder()

    # Небольшая проверка на совпадение размерности
    if encoder.meta.dim != loaded_index.meta.dim:
        raise RuntimeError(
            f"Размерность эмбеддингов ({encoder.meta.dim}) "
            f"не совпадает с размерностью индекса ({loaded_index.meta.dim}). "
            "Скорее всего, индекс был собран другой моделью. "
            "Пересобери индекс через index_build.py."
        )

    # 5) Ретривер (min_score можно потом вынести в конфиг)
    retriever = Retriever(
        encoder,          # EmbeddingEncoder
        loaded_index,     # LoadedIndex
        chunks_by_id,     # {chunk_id -> Chunk}
        min_score=0.2,    # простое значение по умолчанию
    )

    # 6) Генератор ответа (LLM)
    gen_cfg = GenerationConfig(
        model=cfg["generator"]["model"],
        # остальные параметры возьмутся из дефолтов GenerationConfig
    )
    api_key = cfg.get("openai_api_key")
    generator = AnswerGenerator(api_key=api_key, cfg=gen_cfg)

    print("[APP] Пайплайн инициализирован.")
    print(f"[APP] Модель эмбеддингов: {encoder.meta.model_name}")
    print(f"[APP] LLM-модель: {gen_cfg.model}")
    print(f"[APP] Индекс: vectors={loaded_index.meta.num_vectors}, dim={loaded_index.meta.dim}")

    return cfg, retriever, generator


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mini-RAG по локальным документам."
    )
    parser.add_argument(
        "--ask",
        type=str,
        help="Задать вопрос по индексированным документам (CLI-режим).",
    )
    parser.add_argument(
        "--ui",
        action="store_true",
        help="Запустить веб-интерфейс Gradio.",
    )

    args = parser.parse_args()

    try:
        cfg, retriever, generator = init_pipeline()
    except Exception as e:
        print("[APP] Ошибка при инициализации пайплайна:")
        print(e)
        return

    # CLI-режим: python app.py --ask "вопрос"
    if args.ask:
        question = args.ask.strip()
        if not question:
            print("Пожалуйста, введите непустой вопрос.")
            return

        top_k = cfg.get("retriever", {}).get("top_k", 4)

        print("------ ВОПРОС ------")
        print(question)

        retrieval = retriever.search(question, top_k=top_k)
        result = generator.generate(retrieval)

        print("\n------ ОТВЕТ ------")
        print(result.answer)

        print("\n------ ИСТОЧНИКИ ------")
        print(format_citations_markdown(result.citations, show_scores=True))
        return

    # Веб-интерфейс: python app.py --ui
    if args.ui:
        top_k = cfg.get("retriever", {}).get("top_k", 4)

        def answer_ui(question: str) -> tuple[str, str]:
            q = (question or "").strip()
            if not q:
                return "Пожалуйста, введите вопрос.", "Источники:\n—"

            retrieval = retriever.search(q, top_k=top_k)
            result = generator.generate(retrieval)

            answer_md = result.answer
            sources_md = format_citations_markdown(result.citations, show_scores=True)
            return answer_md, sources_md

        with gr.Blocks() as demo:
            gr.Markdown(
                "# mini-rag-local\n"
                "Локальный ассистент по вашим документам (RAG)."
            )

            with gr.Row():
                question_box = gr.Textbox(
                    label="Вопрос",
                    placeholder="Спросите что-нибудь по документам из папки data/...",
                    lines=2,
                )

            answer_md = gr.Markdown(label="Ответ")
            sources_md = gr.Markdown(label="Источники")

            ask_btn = gr.Button("Спросить")

            ask_btn.click(
                answer_ui,
                inputs=question_box,
                outputs=[answer_md, sources_md],
            )
            # Поддержка Enter в текстовом поле
            question_box.submit(
                answer_ui,
                inputs=question_box,
                outputs=[answer_md, sources_md],
            )

        print("[APP] Запуск веб-интерфейса Gradio...")
        demo.launch()
        return

    # Если аргументов нет — просто самотест и краткая сводка
    print("\n[APP] Самотест пройден успешно.")
    print(f"[APP] Число чанков: {len(retriever.chunks_by_id)}")


if __name__ == "__main__":
    main()
