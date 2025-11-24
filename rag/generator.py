from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import openai

from .retriever import RetrievalResult, RetrievedChunk


GenerationStatus = Literal["ok", "insufficient"]


@dataclass
class GenerationConfig:
    model: str # имя модели для API
    max_new_tokens: int = 250 # сколько токенов максимум генерировать
    temperature: float = 0.3 # насколько ответ "творческий" (0.2–0.3 для RAG)
    top_p: float = 0.9 # ещё один регулятор случайности


@dataclass
class GenerationResult:
    status: GenerationStatus
    answer: str # текст ответа
    used_chunks: list[RetrievedChunk] # список чанков, на основе которых отвечали


class AnswerGenerator:
    
    # Обёртка над вызовом LLM.
    def __init__(self, api_key: str, cfg: GenerationConfig) -> None:
        if not api_key or api_key == "your_key_here":
            # Для реальной работы нужно будет подставить реальный ключ
            raise ValueError(
                "OPENAI_API_KEY не задан или равен заглушке. "
                "Укажи реальный ключ в .env."
            )

        self.cfg = cfg
        openai.api_key = api_key

    def _build_system_prompt(self) -> str:
        
        # Системное сообщение — объясняем модели её роль.
        return (
            "Ты ассистент, который отвечает ТОЛЬКО на основе предоставленного контекста.\n"
            "Твои правила:\n"
            "- Используй только факты из контекста ниже.\n"
            "- Если информации недостаточно, честно скажи, что ответить нельзя.\n"
            "- Не придумывай ссылки или источники, которых нет в контексте.\n"
            "- Отвечай кратко и по делу, можно в Markdown."
        )

    def _build_user_prompt(
        self,
        query_text: str,
        chunks: list[RetrievedChunk],
    ) -> str:
        """
        Собираем текст user-промпта: контекст + вопрос.
        """
        if not chunks:
            context_block = "Контекст отсутствует."
        else:
            parts: list[str] = []
            for i, ch in enumerate(chunks, start=1):
                part = (
                    f"[{i}] Файл: {ch.path}\n"
                    f"{ch.text}\n"
                )
                parts.append(part)
            context_block = "\n---\n\n".join(parts)

        prompt = (
            f"Контекст:\n{context_block}\n\n"
            f"Вопрос:\n{query_text}\n\n"
            "Ответь на вопрос, опираясь только на контекст выше. "
            "Если в контексте нет нужной информации, честно скажи, что не можешь "
            "дать точный ответ на основе предоставленных документов."
        )
        return prompt


    def generate(self, retrieval: RetrievalResult) -> GenerationResult:
        """
        Главный метод: по результату ретрива возвращает текст ответа.

        Логика:
        - если status = "insufficient" или чанков нет -> сразу стандартный ответ;
        - иначе -> собираем промпт и вызываем LLM.
        """
        # 1) Если контекста нет — даже не ходим в модель
        if retrieval.status != "ok" or not retrieval.chunks:
            msg = "Данных из источников недостаточно для точного ответа."
            return GenerationResult(
                status="insufficient",
                answer=msg,
                used_chunks=[],
            )

        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(
            query_text=retrieval.query,
            chunks=retrieval.chunks,
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # 2) Вызов LLM через ChatCompletion
        response = openai.ChatCompletion.create(
            model=self.cfg.model,
            messages=messages,
            max_tokens=self.cfg.max_new_tokens,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
        )

        answer_text = (
            response["choices"][0]["message"]["content"].strip()
            if response and response.get("choices")
            else ""
        )

        if not answer_text:
            # На всякий случай fallback — не удалось получить нормальный ответ
            return GenerationResult(
                status="insufficient",
                answer="Не удалось получить ответ от модели.",
                used_chunks=retrieval.chunks,
            )

        return GenerationResult(
            status="ok",
            answer=answer_text,
            used_chunks=retrieval.chunks,
        )


if __name__ == "__main__":

    from pathlib import Path
    from .retriever import RetrievedChunk, RetrievalResult

    print("[GEN] Самотест: сборка промпта и попытка вызова модели (если есть ключ).")

    # Фиктивные чанки
    fake_chunks = [
        RetrievedChunk(
            chunk_id=0,
            path="data/doc1.txt",
            text="RAG — это подход, который сочетает поиск по документам и LLM.",
            snippet="RAG — это подход...",
            index_in_doc=0,
            score=0.9,
        ),
        RetrievedChunk(
            chunk_id=1,
            path="data/doc2.txt",
            text="FAISS используется для быстрого поиска по векторным эмбеддингам.",
            snippet="FAISS используется...",
            index_in_doc=0,
            score=0.85,
        ),
    ]

    retrieval = RetrievalResult(
        status="ok",
        query="Что такое RAG и зачем там FAISS?",
        chunks=fake_chunks,
    )

    # Попробуем прочитать ключ из окружения, чтобы не падать сразу
    import os

    api_key = os.getenv("OPENAI_API_KEY", "")

    if not api_key or api_key == "your_key_here":
        print(
            "[GEN] OPENAI_API_KEY не задан или это заглушка. "
            "Покажем только собранный user-промпт."
        )
        cfg = GenerationConfig(model="gpt-5.1")
        gen = AnswerGenerator.__new__(AnswerGenerator)  # создаём объект без __init__
        gen.cfg = cfg
        # вручную зовём приватный метод для демонстрации
        user_prompt = gen._build_user_prompt(retrieval.query, retrieval.chunks)
        print("------ USER PROMPT ------")
        print(user_prompt)
    else:
        cfg = GenerationConfig(model="gpt-5.1")
        gen = AnswerGenerator(api_key=api_key, cfg=cfg)
        result = gen.generate(retrieval)
        print("------ ANSWER ------")
        print("status:", result.status)
        print(result.answer)
