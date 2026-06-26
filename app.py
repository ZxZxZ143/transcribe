from __future__ import annotations

import csv
import shutil
import threading
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from starlette.staticfiles import StaticFiles

from transcribe import (
    AUDIO_EXTENSIONS,
    load_models,
    transcribe_audio_file,
)
from preprocess import process_item


BASE_DIR = Path("data/api")
UPLOAD_DIR = BASE_DIR / "uploads"
FINAL_CSV_PATH = BASE_DIR / "final_transcripts.csv"

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
]

csv_lock = threading.Lock()

app = FastAPI(title="Transcription API")

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def index():
    return FileResponse("static/index.html")


def prepare_dirs() -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)


def is_allowed_audio(filename: str) -> bool:
    return Path(filename).suffix.lower() in AUDIO_EXTENSIONS


def make_safe_upload_path(original_filename: str) -> Path:
    original_name = Path(original_filename).name
    suffix = Path(original_name).suffix.lower()
    stem = Path(original_name).stem[:80] or "audio"
    return UPLOAD_DIR / f"{stem}_{uuid4().hex}{suffix}"


async def save_upload_file(upload_file: UploadFile) -> Path:
    if not upload_file.filename:
        raise HTTPException(status_code=400, detail="Файл без имени.")

    if not is_allowed_audio(upload_file.filename):
        raise HTTPException(
            status_code=400,
            detail=f"Неподдерживаемый формат файла: {upload_file.filename}",
        )

    file_path = make_safe_upload_path(upload_file.filename)

    with open(file_path, "wb") as output:
        shutil.copyfileobj(upload_file.file, output)

    return file_path


def csv_value(value: Any) -> str | int | float | bool | None:
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return value


def build_csv_row(processed_item: dict[str, Any]) -> dict[str, Any]:
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
        "quality_flags": csv_value(processed_item.get("quality_flags", [])),
        "problem_keywords_found": csv_value(processed_item.get("problem_keywords_found", [])),
        "raw_transcript": processed_item.get("raw_transcript", ""),
        "cleaned_transcript": processed_item.get("cleaned_transcript", ""),
    }


def append_rows_to_csv(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return

    with csv_lock:
        file_exists = FINAL_CSV_PATH.exists() and FINAL_CSV_PATH.stat().st_size > 0

        with open(FINAL_CSV_PATH, "a", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)

            if not file_exists:
                writer.writeheader()

            for row in rows:
                writer.writerow(row)


def transcribe_and_preprocess(audio_path: Path, original_filename: str) -> dict[str, Any]:
    models = app.state.models

    transcript = transcribe_audio_file(
        audio_path=audio_path,
        models=models,
    )

    transcript["file_name"] = original_filename

    processed_item = process_item(transcript)
    return processed_item


@app.on_event("startup")
def startup() -> None:
    prepare_dirs()
    app.state.models = load_models()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/transcripts/upload")
async def upload_transcripts(files: list[UploadFile] = File(...)) -> dict[str, Any]:
    if not files:
        raise HTTPException(status_code=400, detail="Файлы не переданы.")

    response_items = []
    csv_rows = []

    for index, upload_file in enumerate(files, start=1):
        started_at = time.perf_counter()
        original_filename = Path(upload_file.filename or f"audio_{index}").name

        print(f"[{index}/{len(files)}] {original_filename} | processing")

        try:
            saved_audio_path = await save_upload_file(upload_file)
            processed_item = transcribe_and_preprocess(
                audio_path=saved_audio_path,
                original_filename=original_filename,
            )

            csv_rows.append(build_csv_row(processed_item))

            elapsed = time.perf_counter() - started_at
            print(f"[{index}/{len(files)}] {original_filename} | done | {elapsed:.2f}s")

            response_items.append({
                "file_name": original_filename,
                "status": "processed",
                "selected_language": processed_item.get("selected_language"),
                "language_confidence": processed_item.get("language_confidence"),
                "word_count_raw": processed_item.get("word_count_raw"),
                "word_count_cleaned": processed_item.get("word_count_cleaned"),
                "review_priority": processed_item.get("review_priority"),
                "ready_for_classification": processed_item.get("ready_for_classification"),
                "cleaned_transcript": processed_item.get("cleaned_transcript"),
            })

        except HTTPException:
            raise
        except Exception as error:
            elapsed = time.perf_counter() - started_at
            print(f"[{index}/{len(files)}] {original_filename} | error | {elapsed:.2f}s | {error}")

            response_items.append({
                "file_name": original_filename,
                "status": "error",
                "error": str(error),
            })

    append_rows_to_csv(csv_rows)

    return {
        "processed_count": len(csv_rows),
        "total_files": len(files),
        "csv_path": str(FINAL_CSV_PATH),
        "items": response_items,
    }


@app.get("/transcripts/csv")
def download_csv() -> FileResponse:
    if not FINAL_CSV_PATH.exists() or FINAL_CSV_PATH.stat().st_size == 0:
        raise HTTPException(status_code=404, detail="CSV пока не создан.")

    return FileResponse(
        path=FINAL_CSV_PATH,
        media_type="text/csv",
        filename="final_transcripts.csv",
    )
