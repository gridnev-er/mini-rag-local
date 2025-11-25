from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from openai import OpenAI


from .retriever import RetrievalResult, RetrievedChunk
from .citations import SourceCitation, build_citations


GenerationStatus = Literal["ok", "insufficient"]


@dataclass
class GenerationConfig:
    model: str  # имя модели для API
    max_new_tokens: int = 250  # сколько токенов максимум генерировать
    temperature: float = 0.3  # насколько ответ "творческий" (0.2–0.3 для RAG)
    top_p: float = 0.9  # ещё один регулятор случайности


@dataclass
class GenerationResult:
    status: GenerationStatus
    answer: str  # текст ответа
    used_chunks: list[RetrievedChunk]  # список чанков, на основе которых отвечали
    citations: list[SourceCitation] = field(default_factory=list)  # источники для "Источников"


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
        # Новый клиент из openai>=1.0
        self.client = OpenAI(api_key=api_key)


    def _build_system_prompt(self) -> str:
        """
        Системное сообщение — объясняем модели её роль и правила.
        """
        return (
            "Ты ассистент, который в первую очередь отвечает на основе предоставленного контекста.\n"
            "Твои правила:\n"
            "- Используй факты из контекста ниже как основной источник.\n"
            "- Если точной инструкции или формулировки нет, но в контексте есть косвенные намёки, "
            "можешь сделать осторожное предположение и явно отметить в ответе, что это догадка.\n"
            "- Если вообще нет зацепок в контексте, честно скажи, что не можешь ответить на основе "
            "предоставленных документов.\n"
            "- Не придумывай ссылки или источники, которых нет в контексте.\n"
            "- Отвечай кратко и по делу."
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
        По результату ретрива возвращает текст ответа и список источников.
        """

        # 1) Если контекста нет — даже не ходим в модель
        if retrieval.status != "ok" or not retrieval.chunks:
            msg = "Данных из источников недостаточно для точного ответа."
            return GenerationResult(
                status="insufficient",
                answer=msg,
                used_chunks=[],
                citations=[],
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

        # 2) Вызов LLM через новый клиент OpenAI
        response = self.client.chat.completions.create(
            model=self.cfg.model,
            messages=messages,
            # max_tokens в новой библиотеке заменён на max_completion_tokens
            max_completion_tokens=self.cfg.max_new_tokens,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
        )

        answer_text = (
            response.choices[0].message.content.strip()
            if response and response.choices
            else ""
        )


        if not answer_text:
            # На всякий случай fallback — не удалось получить нормальный ответ
            return GenerationResult(
                status="insufficient",
                answer="Не удалось получить ответ от модели.",
                used_chunks=retrieval.chunks,
                citations=[],
            )

        # 3) Строим список источников на основе использованных чанков
        citations = build_citations(retrieval.chunks)

        return GenerationResult(
            status="ok",
            answer=answer_text,
            used_chunks=retrieval.chunks,
            citations=citations,
        )