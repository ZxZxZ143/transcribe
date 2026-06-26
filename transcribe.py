import json
import math
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from vosk import KaldiRecognizer, Model, SetLogLevel

RAW_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/transcripts")
OUTPUT_JSON = OUTPUT_DIR / "transcripts.json"

RU_DETECT_MODEL_PATH = Path("models/vosk-model-small-ru-0.22")
KK_DETECT_MODEL_PATH = Path("models/vosk-model-small-kz-0.42")
RU_TRANSCRIBE_MODEL_PATH = Path("models/vosk-model-ru-0.42")
KK_TRANSCRIBE_MODEL_PATH = Path("models/vosk-model-kz-0.42")

SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2
PCM_CHUNK_BYTES = 16000

DETECTION_WINDOW_RATIO = 0.08
DETECTION_WINDOW_START_RATIOS = [0.25, 0.50, 0.75]
DETECTION_SILENCE_BETWEEN_WINDOWS_SECONDS = 0.25
MIN_DETECTION_WINDOW_SECONDS = 2.5
MAX_DETECTION_WINDOW_SECONDS = 5.0

RU_DEFAULT_BONUS = 0.45
KK_NO_EVIDENCE_PENALTY = 1.35
KK_WEAK_EVIDENCE_PENALTY = 0.55
KAZAKH_MIN_EVIDENCE_TO_SELECT_KK = 2.0
KAZAKH_STRONG_EVIDENCE = 4.0
KAZAKH_STRONG_SCORE_MULTIPLIER = 1.35
KAZAKH_MIN_ABSOLUTE_ADVANTAGE = 0.55

RUN_SHORT_RECHECK_ON_WEAK_KK = True
RUN_FULL_DOUBLE_CHECK_ON_LOW_CONFIDENCE = False
DEFAULT_LANGUAGE_ON_UNKNOWN = "ru"
SAVE_WORDS = True
DETECTION_WORD_TIMESTAMPS = True

AUDIO_EXTENSIONS = {
    ".wav", ".mp3", ".m4a", ".mp4", ".ogg",
    ".flac", ".aac", ".wma", ".webm", ".opus",
    ".aiff", ".aif",
}

KAZAKH_SPECIAL_CHARS = set("әғқңөұүһіӘҒҚҢӨҰҮҺІ")

RUSSIAN_HINT_WORDS = {
    "здравствуйте", "добрый", "день", "вечер", "утро",
    "можно", "скажите", "пожалуйста", "спасибо",
    "да", "нет", "хорошо", "сейчас", "будет",
    "номер", "заявка", "договор", "оплата", "банк",
    "кредит", "карта", "счет", "счёт", "менеджер",
    "оператор", "клиент", "услуга", "подскажите",
}

KAZAKH_HINT_WORDS = {
    "сәлеметсіз", "сәлем", "қайырлы", "рахмет",
    "иә", "жоқ", "жақсы", "қазір", "өтінемін",
    "айтыңыз", "нөмір", "өтініш", "төлем",
    "шот", "несие", "қызмет", "күн", "кеш",
}


def get_audio_files(raw_dir: Path) -> list[Path]:
    if not raw_dir.exists():
        raise FileNotFoundError(f"Папка не найдена: {raw_dir}")

    return sorted(
        file_path
        for file_path in raw_dir.rglob("*")
        if file_path.is_file() and file_path.suffix.lower() in AUDIO_EXTENSIONS
    )


def check_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg не найден. Установи ffmpeg и добавь его в PATH.")


def resolve_model_path(preferred_path: Path, fallback_path: Path) -> Path:
    if preferred_path.exists():
        return preferred_path

    if fallback_path.exists():
        return fallback_path

    raise FileNotFoundError(
        "Не найдена модель. Проверь пути:\n"
        f"  {preferred_path}\n"
        f"  {fallback_path}"
    )


def check_model_path(model_path: Path) -> None:
    if not model_path.exists():
        raise FileNotFoundError(f"Не найдена модель: {model_path}")


def load_models() -> dict[str, Any]:
    ru_detect_path = resolve_model_path(RU_DETECT_MODEL_PATH, RU_TRANSCRIBE_MODEL_PATH)
    kk_detect_path = resolve_model_path(KK_DETECT_MODEL_PATH, KK_TRANSCRIBE_MODEL_PATH)

    check_model_path(RU_TRANSCRIBE_MODEL_PATH)
    check_model_path(KK_TRANSCRIBE_MODEL_PATH)

    cache: dict[str, Model] = {}

    def get_model(model_path: Path) -> Model:
        resolved_path = str(model_path.resolve())
        if resolved_path not in cache:
            cache[resolved_path] = Model(str(model_path))
        return cache[resolved_path]

    return {
        "detect": {
            "ru": get_model(ru_detect_path),
            "kk": get_model(kk_detect_path),
        },
        "transcribe": {
            "ru": get_model(RU_TRANSCRIBE_MODEL_PATH),
            "kk": get_model(KK_TRANSCRIBE_MODEL_PATH),
        },
    }


def decode_audio_to_pcm(audio_path: Path) -> dict[str, Any]:
    check_ffmpeg()

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-i", str(audio_path),
        "-ar", str(SAMPLE_RATE),
        "-ac", "1",
        "-f", "s16le",
        "-acodec", "pcm_s16le",
        "pipe:1",
    ]

    try:
        process = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
    except subprocess.CalledProcessError as error:
        stderr = error.stderr.decode("utf-8", errors="ignore")
        raise RuntimeError(f"ffmpeg не смог декодировать файл: {stderr}") from error

    pcm_bytes = process.stdout

    if not pcm_bytes:
        raise ValueError("После декодирования получился пустой PCM-поток.")

    duration_seconds = len(pcm_bytes) / (SAMPLE_RATE * BYTES_PER_SAMPLE)

    return {
        "pcm_bytes": pcm_bytes,
        "duration_seconds": duration_seconds,
    }


def slice_pcm(pcm_bytes: bytes, start_seconds: float, duration_seconds: float) -> bytes:
    start_byte = int(start_seconds * SAMPLE_RATE) * BYTES_PER_SAMPLE
    end_byte = int((start_seconds + duration_seconds) * SAMPLE_RATE) * BYTES_PER_SAMPLE

    start_byte = max(0, start_byte)
    end_byte = min(len(pcm_bytes), end_byte)

    if start_byte >= len(pcm_bytes):
        return b""

    return pcm_bytes[start_byte:end_byte]


def build_detection_pcm(pcm_bytes: bytes, audio_duration_seconds: float) -> bytes:
    if audio_duration_seconds <= 0:
        raise ValueError("Длительность аудио должна быть больше 0.")

    window_duration_seconds = min(
        max(audio_duration_seconds * DETECTION_WINDOW_RATIO, MIN_DETECTION_WINDOW_SECONDS),
        MAX_DETECTION_WINDOW_SECONDS,
        audio_duration_seconds,
    )

    chunks = []

    for start_ratio in DETECTION_WINDOW_START_RATIOS:
        start_seconds = audio_duration_seconds * start_ratio

        if start_seconds + window_duration_seconds > audio_duration_seconds:
            start_seconds = max(0.0, audio_duration_seconds - window_duration_seconds)

        chunk = slice_pcm(
            pcm_bytes=pcm_bytes,
            start_seconds=start_seconds,
            duration_seconds=window_duration_seconds,
        )

        if chunk:
            chunks.append(chunk)

    if not chunks:
        chunk = slice_pcm(
            pcm_bytes=pcm_bytes,
            start_seconds=0,
            duration_seconds=min(window_duration_seconds, audio_duration_seconds),
        )
        if chunk:
            chunks.append(chunk)

    if not chunks:
        raise ValueError("Не удалось собрать фрагмент для определения языка.")

    silence_bytes = b"\x00\x00" * int(
        SAMPLE_RATE * DETECTION_SILENCE_BETWEEN_WINDOWS_SECONDS
    )

    return silence_bytes.join(chunks)


def recognize_pcm_bytes(pcm_bytes: bytes, model: Model, include_words: bool) -> dict[str, Any]:
    if not pcm_bytes:
        raise ValueError("PCM-буфер пустой.")

    recognizer = KaldiRecognizer(model, SAMPLE_RATE)
    recognizer.SetWords(include_words)

    segments = []
    words = []

    for start in range(0, len(pcm_bytes), PCM_CHUNK_BYTES):
        data = pcm_bytes[start:start + PCM_CHUNK_BYTES]

        if recognizer.AcceptWaveform(data):
            segment = parse_vosk_result(json.loads(recognizer.Result()))
            if segment is not None:
                segments.append(segment)
                words.extend(segment["words"])

    final_segment = parse_vosk_result(json.loads(recognizer.FinalResult()))
    if final_segment is not None:
        segments.append(final_segment)
        words.extend(final_segment["words"])

    text = " ".join(segment["text"] for segment in segments if segment["text"]).strip()

    return {
        "text": text,
        "segments": segments,
        "words": words if include_words else [],
        "word_count": len(words) if include_words else len(tokenize_text(text)),
        "avg_confidence": calculate_average_confidence(words) if include_words else None,
    }


def parse_vosk_result(result: dict[str, Any]) -> dict[str, Any] | None:
    text = result.get("text", "").strip()
    raw_words = result.get("result", [])

    if not text and not raw_words:
        return None

    words = [
        {
            "word": word.get("word", ""),
            "start": round(float(word.get("start", 0.0)), 2),
            "end": round(float(word.get("end", 0.0)), 2),
            "conf": round(float(word.get("conf", 0.0)), 4),
        }
        for word in raw_words
    ]

    return {
        "start": words[0]["start"] if words else None,
        "end": words[-1]["end"] if words else None,
        "text": text,
        "words": words,
    }


def calculate_average_confidence(words: list[dict[str, Any]]) -> float | None:
    confidences = [float(word["conf"]) for word in words if "conf" in word]

    if not confidences:
        return None

    return sum(confidences) / len(confidences)


def tokenize_text(text: str) -> list[str]:
    return re.findall(r"[а-яёәғқңөұүһі]+", text.lower())


def build_text_features(text: str) -> dict[str, Any]:
    text_lower = text.lower()
    tokens = tokenize_text(text_lower)
    unique_words = set(tokens)

    kazakh_chars_count = sum(
        1 for char in text_lower
        if char in KAZAKH_SPECIAL_CHARS
    )

    russian_hint_matches = len(unique_words & RUSSIAN_HINT_WORDS)
    kazakh_hint_matches = len(unique_words & KAZAKH_HINT_WORDS)

    kazakh_evidence_score = min(kazakh_chars_count, 6) + kazakh_hint_matches * 2.0

    return {
        "word_count": len(tokens),
        "kazakh_chars_count": kazakh_chars_count,
        "russian_hint_matches": russian_hint_matches,
        "kazakh_hint_matches": kazakh_hint_matches,
        "kazakh_evidence_score": kazakh_evidence_score,
    }


def analyze_language_result(result: dict[str, Any], language: str) -> dict[str, Any]:
    text = result.get("text", "").lower().strip()
    features = build_text_features(text)
    word_count = features["word_count"]

    if not text or word_count == 0:
        return {"score": 0.0, "features": features}

    avg_confidence = result.get("avg_confidence")
    confidence_factor = avg_confidence if avg_confidence is not None else 0.72
    confidence_factor = max(min(float(confidence_factor), 1.0), 0.05)

    score = min(math.log1p(word_count), math.log1p(22)) * confidence_factor

    kazakh_chars_count = features["kazakh_chars_count"]
    russian_hint_matches = features["russian_hint_matches"]
    kazakh_hint_matches = features["kazakh_hint_matches"]
    kazakh_evidence_score = features["kazakh_evidence_score"]

    if language == "ru":
        score += russian_hint_matches * 0.65

        if kazakh_chars_count == 0:
            score += RU_DEFAULT_BONUS

        score -= min(kazakh_chars_count * 0.12, 0.8)

    elif language == "kk":
        score += min(kazakh_chars_count, 6) * 0.38
        score += kazakh_hint_matches * 0.85

        if kazakh_evidence_score == 0:
            score -= KK_NO_EVIDENCE_PENALTY
        elif kazakh_evidence_score < KAZAKH_MIN_EVIDENCE_TO_SELECT_KK:
            score -= KK_WEAK_EVIDENCE_PENALTY

    return {
        "score": round(max(score, 0.0), 4),
        "features": features,
    }


def choose_language_with_guardrails(
    ru_analysis: dict[str, Any],
    kk_analysis: dict[str, Any],
) -> dict[str, Any]:
    ru_score = ru_analysis["score"]
    kk_score = kk_analysis["score"]

    if ru_score == 0 and kk_score == 0:
        return {
            "selected_language": "unknown",
            "raw_selected_language": "unknown",
            "guardrail_applied": False,
        }

    raw_selected_language = "ru" if ru_score >= kk_score else "kk"
    selected_language = raw_selected_language
    guardrail_applied = False

    if raw_selected_language == "kk":
        kk_evidence = kk_analysis["features"]["kazakh_evidence_score"]
        kk_absolute_advantage = kk_score - ru_score
        kk_score_multiplier_ok = kk_score >= ru_score * KAZAKH_STRONG_SCORE_MULTIPLIER if ru_score > 0 else True

        weak_evidence = kk_evidence < KAZAKH_MIN_EVIDENCE_TO_SELECT_KK
        not_enough_advantage = kk_absolute_advantage < KAZAKH_MIN_ABSOLUTE_ADVANTAGE

        if weak_evidence and (not kk_score_multiplier_ok or not_enough_advantage):
            selected_language = "ru"
            guardrail_applied = True

    return {
        "selected_language": selected_language,
        "raw_selected_language": raw_selected_language,
        "guardrail_applied": guardrail_applied,
    }


def calculate_language_confidence(
    ru_score: float,
    kk_score: float,
    selected_language: str,
    guardrail_applied: bool,
) -> str:
    best_score = max(ru_score, kk_score)
    second_score = min(ru_score, kk_score)
    margin = (best_score - second_score) / best_score if best_score > 0 else 0.0

    if selected_language == "unknown":
        return "low"

    if guardrail_applied:
        return "medium" if margin >= 0.08 else "low"

    if margin >= 0.22:
        return "high"

    if margin >= 0.10:
        return "medium"

    return "low"


def should_run_short_kk_recheck(
    selected_language: str,
    raw_selected_language: str,
    kk_analysis: dict[str, Any],
    confidence: str,
) -> bool:
    if not RUN_SHORT_RECHECK_ON_WEAK_KK:
        return False

    if selected_language != "kk" and raw_selected_language != "kk":
        return False

    kk_evidence = kk_analysis["features"]["kazakh_evidence_score"]

    return kk_evidence < KAZAKH_STRONG_EVIDENCE or confidence in {"low", "medium"}


def detect_main_language(
    pcm_bytes: bytes,
    audio_duration_seconds: float,
    models: dict[str, Any],
) -> dict[str, Any]:
    detection_pcm = build_detection_pcm(
        pcm_bytes=pcm_bytes,
        audio_duration_seconds=audio_duration_seconds,
    )

    ru_probe = recognize_pcm_bytes(
        pcm_bytes=detection_pcm,
        model=models["detect"]["ru"],
        include_words=DETECTION_WORD_TIMESTAMPS,
    )

    kk_probe = recognize_pcm_bytes(
        pcm_bytes=detection_pcm,
        model=models["detect"]["kk"],
        include_words=DETECTION_WORD_TIMESTAMPS,
    )

    ru_analysis = analyze_language_result(ru_probe, "ru")
    kk_analysis = analyze_language_result(kk_probe, "kk")

    choice = choose_language_with_guardrails(
        ru_analysis=ru_analysis,
        kk_analysis=kk_analysis,
    )

    confidence = calculate_language_confidence(
        ru_score=ru_analysis["score"],
        kk_score=kk_analysis["score"],
        selected_language=choice["selected_language"],
        guardrail_applied=choice["guardrail_applied"],
    )

    if should_run_short_kk_recheck(
        selected_language=choice["selected_language"],
        raw_selected_language=choice["raw_selected_language"],
        kk_analysis=kk_analysis,
        confidence=confidence,
    ):
        ru_recheck = recognize_pcm_bytes(
            pcm_bytes=detection_pcm,
            model=models["transcribe"]["ru"],
            include_words=True,
        )

        kk_recheck = recognize_pcm_bytes(
            pcm_bytes=detection_pcm,
            model=models["transcribe"]["kk"],
            include_words=True,
        )

        ru_analysis = analyze_language_result(ru_recheck, "ru")
        kk_analysis = analyze_language_result(kk_recheck, "kk")

        choice = choose_language_with_guardrails(
            ru_analysis=ru_analysis,
            kk_analysis=kk_analysis,
        )

        confidence = calculate_language_confidence(
            ru_score=ru_analysis["score"],
            kk_score=kk_analysis["score"],
            selected_language=choice["selected_language"],
            guardrail_applied=choice["guardrail_applied"],
        )

    selected_language = choice["selected_language"]

    if selected_language == "unknown":
        selected_language = DEFAULT_LANGUAGE_ON_UNKNOWN

    return {
        "language": selected_language,
        "confidence": confidence,
    }


def score_language_result(result: dict[str, Any], language: str) -> float:
    return analyze_language_result(result, language)["score"]


def transcribe_audio_file(audio_path: Path, models: dict[str, Any]) -> dict[str, Any]:
    decoded = decode_audio_to_pcm(audio_path)
    pcm_bytes = decoded["pcm_bytes"]
    audio_duration_seconds = decoded["duration_seconds"]

    detection = detect_main_language(
        pcm_bytes=pcm_bytes,
        audio_duration_seconds=audio_duration_seconds,
        models=models,
    )

    selected_language = detection["language"]
    transcription = None

    if detection["confidence"] == "low" and RUN_FULL_DOUBLE_CHECK_ON_LOW_CONFIDENCE:
        ru_result = recognize_pcm_bytes(
            pcm_bytes=pcm_bytes,
            model=models["transcribe"]["ru"],
            include_words=SAVE_WORDS,
        )

        kk_result = recognize_pcm_bytes(
            pcm_bytes=pcm_bytes,
            model=models["transcribe"]["kk"],
            include_words=SAVE_WORDS,
        )

        if score_language_result(ru_result, "ru") >= score_language_result(kk_result, "kk"):
            selected_language = "ru"
            transcription = ru_result
        else:
            selected_language = "kk"
            transcription = kk_result

    if transcription is None:
        transcription = recognize_pcm_bytes(
            pcm_bytes=pcm_bytes,
            model=models["transcribe"][selected_language],
            include_words=SAVE_WORDS,
        )

    return {
        "file_name": audio_path.name,
        "language": {
            "code": selected_language,
            "confidence": detection["confidence"],
        },
        "text": transcription["text"],
        "segments": [
            {
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"],
            }
            for segment in transcription["segments"]
        ],
        "words": transcription["words"],
        "word_count": transcription["word_count"],
    }


def save_json_safely(data: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(".tmp")

    with open(temp_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

    temp_path.replace(output_path)


def transcribe_folder(
    raw_dir: str | Path = RAW_DIR,
    output_json: str | Path = OUTPUT_JSON,
) -> list[dict[str, Any]]:
    SetLogLevel(-1)

    raw_dir = Path(raw_dir)
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    audio_files = get_audio_files(raw_dir)

    if not audio_files:
        save_json_safely({"results": []}, output_json)
        print(f"no_files | {raw_dir}")
        return []

    models = load_models()
    results = []

    for index, audio_path in enumerate(audio_files, start=1):
        file_start = time.perf_counter()
        print(f"[{index}/{len(audio_files)}] {audio_path.name} | processing")

        try:
            result = transcribe_audio_file(
                audio_path=audio_path,
                models=models,
            )

            results.append(result)
            elapsed = time.perf_counter() - file_start
            print(f"[{index}/{len(audio_files)}] {audio_path.name} | done | {result['language']['code']} | {elapsed:.2f}s")

        except Exception as error:
            elapsed = time.perf_counter() - file_start
            print(f"[{index}/{len(audio_files)}] {audio_path.name} | error | {elapsed:.2f}s | {error}")

        save_json_safely({"results": results}, output_json)

    return results


def transcribe_single_file(
    audio_path: str | Path,
    output_json: str | Path = OUTPUT_JSON,
) -> dict[str, Any]:
    SetLogLevel(-1)

    audio_path = Path(audio_path)
    output_json = Path(output_json)

    if not audio_path.exists():
        raise FileNotFoundError(f"Файл не найден: {audio_path}")

    if not audio_path.is_file():
        raise ValueError(f"Это не файл: {audio_path}")

    if audio_path.suffix.lower() not in AUDIO_EXTENSIONS:
        raise ValueError(f"Неподдерживаемый формат аудио: {audio_path.suffix}")

    file_start = time.perf_counter()
    print(f"{audio_path.name} | processing")

    try:
        models = load_models()
        result = transcribe_audio_file(
            audio_path=audio_path,
            models=models,
        )

        save_json_safely(result, output_json)

        elapsed = time.perf_counter() - file_start
        print(f"{audio_path.name} | done | {result['language']['code']} | {elapsed:.2f}s")

        return result

    except Exception as error:
        elapsed = time.perf_counter() - file_start
        print(f"{audio_path.name} | error | {elapsed:.2f}s | {error}")
        raise



def main() -> None:
    total_start = time.perf_counter()
    results = transcribe_folder(RAW_DIR, OUTPUT_JSON)
    elapsed = time.perf_counter() - total_start
    print(f"ready | files: {len(results)} | {elapsed:.2f}s")


if __name__ == "__main__":
    main()
