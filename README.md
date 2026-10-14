# Speech & Transcript Analytics

A local application for **Russian and Kazakh audio transcription**, transcript quality review and semantic clustering. A FastAPI backend serves the browser interface, manages uploads and CSV results, and connects speech processing to embedding-based analysis.

**Stack:** Python · FastAPI · Vosk · FFmpeg · pandas · BGE-M3 · scikit-learn

[Pipeline](#pipeline) · [Quick start](#quick-start) · [API](#api) · [Limits](#current-status-and-limitations)

## Pipeline

```text
Audio upload → FFmpeg mono PCM at 16 kHz
             → RU/KK Vosk language scoring and recognition
             → raw transcript
             → deterministic cleanup, quality flags and review priority
             → cleaned transcript / semantic text / CSV
             → BGE-M3 embeddings → optional PCA → clustering
             → browser results / cluster CSV / summary report
```

- [`transcription/transcribe.py`](transcription/transcribe.py) samples audio windows, scores recognition evidence in both languages and chooses the transcription model. Language confidence is a heuristic, not calibrated probability.
- [`transcription/preprocess.py`](transcription/preprocess.py) retains raw, cleaned and semantic text separately, marks empty/low-quality calls and assigns review flags.
- [`clustering/cluster_transcripts.py`](clustering/cluster_transcripts.py) supports agglomerative clustering, K-means and HDBSCAN with per-stage timing diagnostics. The API selects agglomerative clustering with nine requested clusters and PCA with twenty requested components; the implementation adapts to available data.
- [`clustering/cluster_report.py`](clustering/cluster_report.py) builds inspectable cluster summaries.

## Quick start

Run all commands from the repository root. Install Python and [FFmpeg](https://www.ffmpeg.org/download.html), making `ffmpeg` available on `PATH`.

```bash
git clone https://github.com/ZxZxZ143/transcribe.git
cd transcribe
python -m venv .venv
```

Install dependencies using the virtual environment's Python:

```powershell
# Windows PowerShell
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

```bash
# macOS / Linux
./.venv/bin/python -m pip install -r requirements.txt
```

Download and extract the Russian and Kazakh models from the [official Vosk model catalog](https://alphacephei.com/vosk/models) into these exact directories:

```text
models/vosk-model-ru-0.42/
models/vosk-model-kz-0.42/
```

The optional small detection models are `models/vosk-model-small-ru-0.22/` and `models/vosk-model-small-kz-0.42/`. When those are absent, detection reuses the corresponding full transcription model. The full RU and KK models remain required. Startup fails visibly if model directories are missing.

Start a loopback-only server:

```powershell
& .\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000
```

On macOS/Linux, use `./.venv/bin/python` for the same command. Open `http://127.0.0.1:8000` for the UI or `/docs` for OpenAPI. Upload a synthetic or authorized recording, inspect the transcript/quality fields, then run clustering after collecting usable transcripts. The first clustering call may download [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3); provision model access and sufficient memory.

## API

| Endpoint | Purpose |
| --- | --- |
| `GET /api/health` | Application health |
| `POST /api/upload` | Process uploaded audio files |
| `GET /api/results` | Read transcript results |
| `GET /api/csv` | Export transcript CSV |
| `POST /api/clustering/run` | Cluster collected transcripts |
| `GET /api/clustering/results` | Read cluster assignments |
| `GET /api/clustering/csv` | Export cluster CSV |
| `DELETE /api/results` | Clear local results and uploaded audio |

See [`app.py`](app.py) for request parameters and legacy endpoint aliases. The delete endpoint removes local material; use it only when those results are disposable.

Standalone clustering and reporting use explicit paths, avoiding working-directory-dependent defaults:

```bash
python -m clustering.cluster_transcripts --input-csv data/api/final_transcripts.csv --output-csv data/clustering/clustered_transcripts.csv --algorithm agglomerative --reduce-dim pca
python -m clustering.cluster_report --clustered-csv data/clustering/clustered_transcripts.csv --report-csv data/clustering/clusters_summary.csv
```

Use your virtual environment's Python for these commands.

## Current status and limitations

This is an ML application prototype with a browser UI. There is no repository test suite, deployment pipeline or published accuracy benchmark. Model quality, language scoring and cluster usefulness require evaluation on representative authorized data. Cluster numbers do not imply validated semantic classes.

The implemented speech path uses **Vosk**. `faster-whisper` is listed in requirements but is not the active recognizer in `transcription/transcribe.py`. Dependency versions are unpinned, and model weights are separately provisioned.

The service uses process-local models and locks, local uploaded audio and CSV storage. It has no authentication, production upload quotas or distributed job queue. Keep it bound to loopback for local evaluation. Review retention and sensitive transcript handling before any wider use; do not commit recordings, transcripts or runtime exports.
