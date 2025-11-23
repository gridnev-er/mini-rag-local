from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv


def _load_yaml(config_path: Path) -> Dict[str, Any]:
    
    #Бросает понятную ошибку, если файла нет или он сломан.
    if not config_path.exists():
        raise FileNotFoundError(
            f"Файл конфигурации {config_path} не найден. "
            "Создай его или скопируй пример."
        )

    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError(
            f"Ожидаю, что {config_path} содержит YAML-объект верхнего уровня."
        )

    return data


def _apply_defaults(raw: Dict[str, Any]) -> Dict[str, Any]:

    # Подмешивает дефолты к тому, что пришло из YAML.
    # Возвращает уже нормализованный словарь config.

    # --- paths ---
    raw_paths = raw.get("paths", {}) or {}
    data_dir = Path(raw_paths.get("data_dir", "./data"))
    index_dir = Path(raw_paths.get("index_dir", "./index"))

    # --- chunking ---
    raw_chunking = raw.get("chunking", {}) or {}
    chunk_size = int(raw_chunking.get("chunk_size", 1000))
    overlap = int(raw_chunking.get("overlap", 150))

    # --- retriever ---
    raw_retriever = raw.get("retriever", {}) or {}
    top_k = int(raw_retriever.get("top_k", 4))

    # --- generator ---
    raw_generator = raw.get("generator", {}) or {}
    mode = str(raw_generator.get("mode", "api"))
    model = str(raw_generator.get("model", "gpt-4o-mini"))
    max_new_tokens = int(raw_generator.get("max_new_tokens", 250))
    temperature = float(raw_generator.get("temperature", 0.3))
    top_p = float(raw_generator.get("top_p", 0.9))

    cfg: Dict[str, Any] = {
        "paths": {
            "data_dir": data_dir,
            "index_dir": index_dir,
        },
        "chunking": {
            "chunk_size": chunk_size,
            "overlap": overlap,
        },
        "retriever": {
            "top_k": top_k,
        },
        "generator": {
            "mode": mode,
            "model": model,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
        },
        "openai_api_key": None,
    }
    return cfg


def _validate_config(cfg: Dict[str, Any]) -> None:
    
    #Проверяет базовые вещи и даёт человекочитаемые ошибки.
    paths = cfg["paths"]
    chunking = cfg["chunking"]
    retriever = cfg["retriever"]
    generator = cfg["generator"]

    data_dir: Path = paths["data_dir"]
    index_dir: Path = paths["index_dir"]

    # Пути
    if not data_dir.exists():
        raise RuntimeError(
            f"Каталог с документами {data_dir} не существует. "
            "Создай папку или поправь paths.data_dir в config.yaml."
        )

    # index_dir можно создать автоматически
    index_dir.mkdir(parents=True, exist_ok=True)

    # Чанкинг
    chunk_size = chunking["chunk_size"]
    overlap = chunking["overlap"]

    if chunk_size <= 0:
        raise ValueError("chunk_size должен быть > 0.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap должен быть в диапазоне 0 ≤ overlap < chunk_size."
        )

    # Ретривер
    top_k = retriever["top_k"]
    if top_k < 1:
        raise ValueError("retriever.top_k должен быть ≥ 1.")

    # Генератор + ключ
    mode = generator["mode"]
    if mode not in ("api",):
        raise ValueError(
            f"generator.mode={mode!r} пока не поддерживается. Используй 'api'."
        )

    api_key = cfg.get("openai_api_key")
    if mode == "api" and not api_key:
        raise RuntimeError(
            "Режим generator.mode='api', но не найден OPENAI_API_KEY. "
            "Создай .env по примеру .env.example и укажи ключ."
        )


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Главная точка входа.

    1. Загружает переменные окружения из .env
    2. Читает config.yaml
    3. Подмешивает дефолты
    4. Подставляет OPENAI_API_KEY (если есть)
    5. Валидирует конфиг
    6. Возвращает словарь config
    """

    load_dotenv()
    raw_yaml = _load_yaml(Path(config_path))
    cfg = _apply_defaults(raw_yaml)
    api_key = os.getenv("OPENAI_API_KEY")

    if api_key:
        cfg["openai_api_key"] = api_key

    _validate_config(cfg)

    return cfg
