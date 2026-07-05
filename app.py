from __future__ import annotations

import csv
import shutil
import threading
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response
from starlette.staticfiles import StaticFiles

from transcription.transcribe import (
    AUDIO_EXTENSIONS,
    load_models,
    transcribe_audio_file,
)
from transcription.preprocess import process_item


BASE_DIR = Path("data/api")
UPLOAD_DIR = BASE_DIR / "uploads"
FINAL_CSV_PATH = BASE_DIR / "final_transcripts.csv"

CLUSTERING_DIR = Path("data/clustering")
CLUSTERED_CSV_PATH = CLUSTERING_DIR / "clustered_transcripts.csv"

STATIC_DIR = Path("static")
INDEX_HTML_PATH = STATIC_DIR / "index.html"

CSV_COLUMNS = [
    "file_name",
    "status",
    "selected_language",
    "language_confidence",
    "word_count_raw",
    "word_count_cleaned",
    "is_empty_call",
    "empty_reason",
    "review_priority",
    "ready_for_classification",
    "quality_flags",
    "problem_keywords_found",
    "raw_transcript",
    "cleaned_transcript",
    "semantic_text",
]

csv_lock = threading.Lock()
clustering_lock = threading.Lock()

app = FastAPI(title="Transcription API")

app.mount(
    "/static",
    StaticFiles(directory=str(STATIC_DIR), check_dir=False),
    name="static",
)


@app.on_event("startup")
def startup() -> None:
    prepare_dirs()
    app.state.models = load_models()


@app.get("/", response_model=None)
def index():
    if INDEX_HTML_PATH.exists():
        return FileResponse(INDEX_HTML_PATH)

    return HTMLResponse(
        """
        <html>
            <head>
                <title>Transcription API</title>
            </head>
            <body style="font-family: Arial; padding: 32px;">
                <h1>Transcription API is running</h1>
                <p>Frontend file was not found: static/index.html</p>
                <p>Open API docs: <a href="/docs">/docs</a></p>
            </body>
        </html>
        """
    )


@app.get("/favicon.ico")
def favicon() -> Response:
    return Response(status_code=204)


def prepare_dirs() -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    CLUSTERING_DIR.mkdir(parents=True, exist_ok=True)


def is_allowed_audio(filename: str) -> bool:
    return Path(filename).suffix.lower() in AUDIO_EXTENSIONS


def make_safe_upload_path(original_filename: str) -> Path:
    original_name = Path(original_filename).name
    suffix = Path(original_name).suffix.lower()
    stem = Path(original_name).stem[:80] or "audio"
    return UPLOAD_DIR / f"{stem}_{uuid4().hex}{suffix}"


async def save_upload_file(upload_file: UploadFile) -> Path:
    if not upload_file.filename:
        raise ValueError("Файл без имени.")

    if not is_allowed_audio(upload_file.filename):
        raise ValueError(f"Неподдерживаемый формат файла: {upload_file.filename}")

    file_path = make_safe_upload_path(upload_file.filename)

    with open(file_path, "wb") as output:
        shutil.copyfileobj(upload_file.file, output)

    return file_path


def list_to_csv_text(value: Any) -> Any:
    if value is None:
        return ""

    if isinstance(value, list):
        return ", ".join(str(item) for item in value)

    return value


def build_csv_row(processed_item: dict[str, Any]) -> dict[str, Any]:

    return {
        "file_name": processed_item.get("file_name", ""),
        "status": processed_item.get("status", ""),
        "selected_language": processed_item.get("selected_language", ""),
        "language_confidence": processed_item.get("language_confidence", ""),
        "word_count_raw": processed_item.get("word_count_raw", ""),
        "word_count_cleaned": processed_item.get("word_count_cleaned", ""),
        "is_empty_call": processed_item.get("is_empty_call", ""),
        "empty_reason": processed_item.get("empty_reason", ""),
        "review_priority": processed_item.get("review_priority", ""),
        "ready_for_classification": processed_item.get("ready_for_classification", ""),
        "quality_flags": list_to_csv_text(processed_item.get("quality_flags", [])),
        "problem_keywords_found": list_to_csv_text(
            processed_item.get("problem_keywords_found", [])
        ),
        "raw_transcript": processed_item.get("raw_transcript", ""),
        "cleaned_transcript": processed_item.get("cleaned_transcript", ""),
        "semantic_text": processed_item.get("semantic_text", ""),
    }


def ensure_csv_schema() -> None:
    """
    Если CSV был создан старым кодом с неправильным порядком колонок,
    мы не дописываем в него новые строки, а сохраняем старый файл как backup.
    Это защищает от ситуации, когда заголовки одни, а значения пишутся в другие колонки.
    """

    if not FINAL_CSV_PATH.exists() or FINAL_CSV_PATH.stat().st_size == 0:
        return

    try:
        with open(FINAL_CSV_PATH, "r", encoding="utf-8-sig", newline="") as file:
            reader = csv.reader(file)
            existing_columns = next(reader, [])
    except Exception:
        existing_columns = []

    if existing_columns == CSV_COLUMNS:
        return

    backup_path = FINAL_CSV_PATH.with_name(
        f"{FINAL_CSV_PATH.stem}_{int(time.time())}.bak.csv"
    )

    FINAL_CSV_PATH.replace(backup_path)
    print(
        "CSV schema mismatch. "
        f"Old CSV moved to backup: {backup_path}. "
        "A new CSV will be created."
    )


def append_rows_to_csv(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return

    with csv_lock:
        ensure_csv_schema()

        file_exists = FINAL_CSV_PATH.exists() and FINAL_CSV_PATH.stat().st_size > 0

        with open(FINAL_CSV_PATH, "a", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=CSV_COLUMNS,
                extrasaction="ignore",
            )

            if not file_exists:
                writer.writeheader()

            for row in rows:
                correct_row = {column: row.get(column, "") for column in CSV_COLUMNS}
                writer.writerow(correct_row)


def read_csv_results(limit: int = 50) -> list[dict[str, Any]]:
    if not FINAL_CSV_PATH.exists() or FINAL_CSV_PATH.stat().st_size == 0:
        return []

    with csv_lock:
        with open(FINAL_CSV_PATH, "r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            rows = list(reader)

    if limit <= 0:
        return rows

    return rows[-limit:][::-1]


def clear_csv_and_uploads() -> None:
    with csv_lock:
        if FINAL_CSV_PATH.exists():
            FINAL_CSV_PATH.unlink()

    with clustering_lock:
        if CLUSTERED_CSV_PATH.exists():
            CLUSTERED_CSV_PATH.unlink()

    if UPLOAD_DIR.exists():
        for item in UPLOAD_DIR.iterdir():
            if item.is_file():
                item.unlink()


def transcribe_and_preprocess(
    audio_path: Path,
    original_filename: str,
) -> dict[str, Any]:
    models = app.state.models

    transcript = transcribe_audio_file(
        audio_path=audio_path,
        models=models,
    )

    transcript["file_name"] = original_filename

    processed_item = process_item(transcript)

    # На всякий случай фиксируем имя файла уже после process_item.
    processed_item["file_name"] = original_filename

    return processed_item


def serialize_processed_item_for_response(
    processed_item: dict[str, Any],
) -> dict[str, Any]:
    return {
        "file_name": processed_item.get("file_name"),
        "status": processed_item.get("status"),
        "selected_language": processed_item.get("selected_language"),
        "language_confidence": processed_item.get("language_confidence"),
        "word_count_raw": processed_item.get("word_count_raw"),
        "word_count_cleaned": processed_item.get("word_count_cleaned"),
        "is_empty_call": processed_item.get("is_empty_call"),
        "empty_reason": processed_item.get("empty_reason"),
        "review_priority": processed_item.get("review_priority"),
        "ready_for_classification": processed_item.get("ready_for_classification"),
        "quality_flags": processed_item.get("quality_flags", []),
        "problem_keywords_found": processed_item.get("problem_keywords_found", []),
        "raw_transcript": processed_item.get("raw_transcript", ""),
        "cleaned_transcript": processed_item.get("cleaned_transcript", ""),
        "semantic_text": processed_item.get("semantic_text", ""),
    }


def build_error_response_item(
    filename: str,
    error: Exception | str,
) -> dict[str, Any]:
    return {
        "file_name": filename,
        "status": "error",
        "error": str(error),
        "selected_language": "",
        "language_confidence": "",
        "word_count_raw": "",
        "word_count_cleaned": "",
        "is_empty_call": True,
        "empty_reason": "processing_error",
        "review_priority": "high",
        "ready_for_classification": False,
        "quality_flags": ["processing_error"],
        "problem_keywords_found": [],
        "raw_transcript": "",
        "cleaned_transcript": "",
        "semantic_text": "",
    }


def parse_csv_value(value: Any) -> Any:
    if value is None:
        return ""

    if not isinstance(value, str):
        return value

    value = value.strip()

    if value == "":
        return ""

    lowered = value.lower()

    if lowered == "true":
        return True

    if lowered == "false":
        return False

    if lowered in {"nan", "none", "null"}:
        return ""

    return value


def read_csv_file_as_records(csv_path: Path) -> list[dict[str, Any]]:
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return []

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        rows = []

        for row in reader:
            cleaned_row = {
                key: parse_csv_value(value)
                for key, value in row.items()
                if key is not None
            }
            rows.append(cleaned_row)

    return rows


def get_cluster_id(row: dict[str, Any]) -> int:
    value = row.get("cluster_id", -1)

    try:
        return int(float(value))
    except (TypeError, ValueError):
        return -1


def get_clusters_count(rows: list[dict[str, Any]]) -> int:
    cluster_ids = {
        get_cluster_id(row)
        for row in rows
        if get_cluster_id(row) != -1
    }

    return len(cluster_ids)


def get_noise_count(rows: list[dict[str, Any]]) -> int:
    return sum(1 for row in rows if get_cluster_id(row) == -1)


def get_cluster_sizes(rows: list[dict[str, Any]]) -> dict[str, int]:
    sizes: dict[str, int] = {}

    for row in rows:
        cluster_id = str(get_cluster_id(row))
        sizes[cluster_id] = sizes.get(cluster_id, 0) + 1

    return dict(sorted(sizes.items(), key=lambda item: int(item[0])))


def read_clustering_results() -> list[dict[str, Any]]:
    with clustering_lock:
        return read_csv_file_as_records(CLUSTERED_CSV_PATH)


def run_clustering_pipeline() -> dict[str, Any]:
    if not FINAL_CSV_PATH.exists() or FINAL_CSV_PATH.stat().st_size == 0:
        raise HTTPException(
            status_code=404,
            detail="final_transcripts.csv пока не создан. Сначала загрузи и обработай аудио.",
        )

    try:
        from clustering.cluster_transcripts import run_clustering
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "Не удалось импортировать run_clustering из "
                "clustering.cluster_transcripts. Проверь, что файл кластеризации "
                "лежит в clustering/cluster_transcripts.py."
            ),
        ) from error

    with clustering_lock:
        result = run_clustering(
            input_csv=FINAL_CSV_PATH,
            output_csv=CLUSTERED_CSV_PATH,
            algorithm="agglomerative",
            n_clusters=9,
            reduce_dim="pca",
            pca_components=20,
            show_progress_bar=False,
        )

    return result


@app.get("/api/health")
@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "models_loaded": hasattr(app.state, "models"),
        "csv_path": str(FINAL_CSV_PATH),
        "clustered_csv_path": str(CLUSTERED_CSV_PATH),
    }


@app.post("/api/upload")
@app.post("/transcripts/upload")
async def upload_transcripts(files: list[UploadFile] = File(...)) -> dict[str, Any]:
    if not files:
        raise HTTPException(status_code=400, detail="Файлы не переданы.")

    response_items: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []

    for index, upload_file in enumerate(files, start=1):
        started_at = time.perf_counter()
        original_filename = Path(upload_file.filename or f"audio_{index}").name

        print(f"[{index}/{len(files)}] {original_filename} | processing")

        try:
            if not upload_file.filename:
                raise ValueError("Файл без имени.")

            if not is_allowed_audio(upload_file.filename):
                raise ValueError(
                    f"Неподдерживаемый формат файла: {upload_file.filename}"
                )

            saved_audio_path = await save_upload_file(upload_file)

            processed_item = transcribe_and_preprocess(
                audio_path=saved_audio_path,
                original_filename=original_filename,
            )

            csv_row = build_csv_row(processed_item)
            csv_rows.append(csv_row)

            elapsed = time.perf_counter() - started_at
            print(
                f"[{index}/{len(files)}] "
                f"{original_filename} | done | {elapsed:.2f}s"
            )

            response_items.append(
                serialize_processed_item_for_response(processed_item)
            )

        except Exception as error:
            elapsed = time.perf_counter() - started_at
            print(
                f"[{index}/{len(files)}] "
                f"{original_filename} | error | {elapsed:.2f}s | {error}"
            )

            response_items.append(
                build_error_response_item(
                    filename=original_filename,
                    error=error,
                )
            )

    append_rows_to_csv(csv_rows)

    processed_count = sum(
        1 for item in response_items
        if item.get("status") == "processed"
    )
    error_count = sum(
        1 for item in response_items
        if item.get("status") == "error"
    )

    return {
        "processed_count": processed_count,
        "error_count": error_count,
        "total_files": len(files),
        "csv_path": str(FINAL_CSV_PATH),
        "results": response_items,
        "items": response_items,
    }


@app.get("/api/results")
@app.get("/transcripts/results")
def get_results(
    limit: int = Query(default=50, ge=1, le=1000),
) -> dict[str, Any]:
    rows = read_csv_results(limit=limit)

    return {
        "count": len(rows),
        "csv_path": str(FINAL_CSV_PATH),
        "results": rows,
    }


@app.post("/api/clustering/run")
def run_clustering_endpoint() -> dict[str, Any]:
    result = run_clustering_pipeline()
    rows = read_clustering_results()

    return {
        "status": "done",
        "input_csv": str(FINAL_CSV_PATH),
        "output_csv": str(CLUSTERED_CSV_PATH),
        "total_rows": len(rows),
        "clusters_count": get_clusters_count(rows),
        "noise_rows": get_noise_count(rows),
        "cluster_sizes": get_cluster_sizes(rows),
        "clustering_result": result,
        "results": rows,
        "items": rows,
    }


@app.get("/api/clustering/results")
def get_clustering_results() -> dict[str, Any]:
    rows = read_clustering_results()

    return {
        "count": len(rows),
        "csv_path": str(CLUSTERED_CSV_PATH),
        "clusters_count": get_clusters_count(rows),
        "noise_rows": get_noise_count(rows),
        "cluster_sizes": get_cluster_sizes(rows),
        "results": rows,
        "items": rows,
    }


@app.get("/api/clustering/csv", response_model=None)
def download_clustering_csv():
    if not CLUSTERED_CSV_PATH.exists() or CLUSTERED_CSV_PATH.stat().st_size == 0:
        raise HTTPException(status_code=404, detail="CSV с кластерами пока не создан.")

    return FileResponse(
        path=CLUSTERED_CSV_PATH,
        media_type="text/csv",
        filename="clustered_transcripts.csv",
    )


@app.get("/api/csv", response_model=None)
@app.get("/api/download/csv", response_model=None)
@app.get("/transcripts/csv", response_model=None)
def download_csv():
    if not FINAL_CSV_PATH.exists() or FINAL_CSV_PATH.stat().st_size == 0:
        raise HTTPException(status_code=404, detail="CSV пока не создан.")

    return FileResponse(
        path=FINAL_CSV_PATH,
        media_type="text/csv",
        filename="final_transcripts.csv",
    )


@app.delete("/api/results")
@app.delete("/transcripts/results")
def clear_results() -> dict[str, Any]:
    clear_csv_and_uploads()

    return {
        "status": "cleared",
        "message": "CSV, кластеры и временные загруженные файлы очищены.",
    }