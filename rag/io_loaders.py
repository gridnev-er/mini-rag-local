from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from pypdf import PdfReader


@dataclass
class Document:

    path: Path  # относительный путь внутри data_dir
    kind: str  # тип файла ("txt" | "md" | "pdf")
    text: str  # нормализованный текст документа
    num_chars: int  # длина текста в символах
    file_size: int  # размер исходного файла в байтах

SUPPORTED_EXTS = {".txt", ".md", ".pdf"}


def _is_hidden_or_junk(path: Path) -> bool:
    
    # Фильтруем скрытые и явно мусорные файлы/папки:
    name = path.name
    if name.startswith("."):
        return True
    if name.startswith("~$"):
        return True
    return False


def _iter_candidate_files(data_dir: Path) -> Iterable[Path]:

    # Рекурсивно обходит data_dir и отдаёт файлы с нужными расширениями.
    for path in data_dir.rglob("*"):
        if not path.is_file():
            continue

        if any(part.startswith(".") for part in path.parts):
            continue
        if _is_hidden_or_junk(path):
            continue

        ext = path.suffix.lower()
        if ext in SUPPORTED_EXTS:
            yield path


def _read_text_file(path: Path) -> str:

    try:
        # errors="ignore" — чтобы странные символы не завалили процесс
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] Не удалось прочитать текстовый файл {path}: {e}")
        return ""


def _read_pdf_file(path: Path) -> str:

    try:
        reader = PdfReader(str(path))
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] Не удалось открыть PDF {path}: {e}")
        return ""

    chunks: List[str] = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception as e:  # noqa: BLE001
            print(f"[WARN] Не удалось извлечь текст со страницы {i} в {path}: {e}")
            text = ""
        if text:
            chunks.append(text)

    return "\n".join(chunks)


def _normalize_text(raw: str) -> str:

    if not raw:
        return ""

    # единый формат переводов строк
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    # убираем BOM и подобное
    text = text.replace("\ufeff", "")

    lines = text.split("\n")
    normalized_lines: List[str] = []
    previous_blank = False

    for line in lines:
        stripped = line.strip()

        # считаем строки длины 0–1 как "шум" и воспринимаем как пустые
        if len(stripped) <= 1:
            is_blank = True
        else:
            is_blank = False

        if is_blank:
            if not previous_blank:
                # добавляем одну пустую строку вместо серии
                normalized_lines.append("")
            previous_blank = True
        else:
            normalized_lines.append(stripped)
            previous_blank = False

    # убираем пустые строки в начале и в конце
    while normalized_lines and normalized_lines[0] == "":
        normalized_lines.pop(0)
    while normalized_lines and normalized_lines[-1] == "":
        normalized_lines.pop()

    return "\n".join(normalized_lines)


def load_documents(data_dir: Path | str) -> List[Document]:

    # Принимает путь к каталогу с документами, возвращает список Document с нормализованным текстом.

    data_dir = Path(data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(
            f"Каталог с документами {data_dir} не существует. "
            "Проверь paths.data_dir в config.yaml."
        )

    documents: List[Document] = []
    total_files = 0
    skipped_files = 0

    for file_path in _iter_candidate_files(data_dir):
        total_files += 1
        ext = file_path.suffix.lower()

        if ext in {".txt", ".md"}:
            raw_text = _read_text_file(file_path)
            kind = ext.lstrip(".")
        elif ext == ".pdf":
            raw_text = _read_pdf_file(file_path)
            kind = "pdf"
        else:
            continue

        normalized = _normalize_text(raw_text)

        # если текст пустой или слишком маленький — пропускаем
        if not normalized or len(normalized) < 5:
            print(f"[INFO] Файл {file_path} пропущен: пустой или слишком короткий текст.")
            skipped_files += 1
            continue

        rel_path = file_path.relative_to(data_dir)
        stat = file_path.stat()

        doc = Document(
            path=rel_path,
            kind=kind,
            text=normalized,
            num_chars=len(normalized),
            file_size=stat.st_size,
        )
        documents.append(doc)

    print(
        f"[INGEST] Обработано файлов: {total_files}, "
        f"создано документов: {len(documents)}, "
        f"пропущено: {skipped_files}."
    )

    return documents


if __name__ == "__main__":
    # Небольшой ручной тест:
    # можно запустить `python -m rag.io_loaders` из корня проекта,
    # чтобы проверить, как он читает файлы из ./data.
    docs = load_documents("./data")
    print(f"Загружено документов: {len(docs)}")
    if docs:
        first = docs[0]
        print("--- Пример документа ---")
        print("path:", first.path)
        print("kind:", first.kind)
        print("num_chars:", first.num_chars)
        print("preview:")
        print(first.text[:300], "...")
