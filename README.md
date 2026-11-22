# AI-Assisted Image Annotation Platform

A web application that makes labeling images for object detection faster by letting computer-vision models do part of the work: a fine-tuned **YOLO11** detector proposes boxes, and **MobileSAM** turns a single click into an object outline. The user accepts, corrects or rejects — and exports a standard **COCO** or **YOLO** dataset.

Built as the project for the *AI Vision* module (SRH Hochschule Heidelberg). Demo use case: labeling **construction-site helmet compliance** (`helmet` / `head`).

> 🚧 Work in progress — see [docs/PROJECT_PLAN.md](docs/PROJECT_PLAN.md).

## Quick start

Requirements: Docker Desktop (or Docker Engine + Compose v2).

```bash
docker compose up --build
```

| Service | URL |
|---|---|
| Web app | http://localhost:3000 |
| API docs (Swagger) | http://localhost:3000/api/docs |
| SeaweedFS filer UI (stored images) | http://localhost:8888 |

Default local credentials are in [.env.example](.env.example); copy it to `.env` to change them.

### AI models

The backend loads two models in the background at startup (the UI shows when they are ready):

| Model | Purpose | Where it comes from |
|---|---|---|
| Fine-tuned **YOLO11** (OpenVINO) | ✨ Suggest: pre-labels helmets / heads | Train with [the Colab notebook](docs/training.md), then put `yolo11n_helmet_openvino_model/` into `weights/` |
| **MobileSAM** | 🪄 Magic click: click an object → outline | Downloaded automatically into `weights/` on first start |

`weights/` is git-ignored and mounted into the backend container. Use another detector file with
e.g. `DETECTOR_WEIGHTS=yolo11s_helmet_openvino_model docker compose up`.

## Architecture

```
Browser ──► frontend (nginx) ──/api──► backend (FastAPI)
                                         ├──► PostgreSQL + pgvector
                                         ├──► SeaweedFS (S3-compatible image storage)
                                         └──► models (YOLO11 / MobileSAM, CPU)
```

## Development

```bash
# Backend (Python 3.11, uv)
docker compose up -d postgres storage   # API tests use a real PostgreSQL (host port 5433)
cd backend
uv sync
uv run pytest                            # creates and migrates the annotator_test database
uv run ruff check . && uv run ruff format --check . && uv run mypy app
uv run alembic revision --autogenerate -m "describe change"   # after changing app/models.py

# Frontend (Node)
cd frontend
npm ci
npm run dev        # http://localhost:5173, proxies /api to localhost:8000
npm run lint && npm run build
```

Install the git hooks once: `uvx pre-commit install`.

The database is exposed on host port **5433** (not 5432), so it does not clash with a locally
installed PostgreSQL.

## API overview

Interactive docs: http://localhost:3000/api/docs

| Method & path | Purpose |
|---|---|
| `GET /api/health` | Database and storage status |
| `POST/GET /api/projects`, `GET/PATCH/DELETE /api/projects/{id}` | Projects and their label classes |
| `POST /api/projects/{id}/images` | Upload images (validated by content: PNG/JPEG/WEBP, size limit) |
| `GET /api/projects/{id}/images?status=` | List images with annotation counts |
| `GET/PATCH/DELETE /api/images/{id}`, `GET /api/images/{id}/file` | Image details, status, file stream |
| `GET/PUT /api/images/{id}/annotations` | Load / replace all annotations of an image (boxes, polygons) |
| `GET/POST /api/models`, `POST /api/models/{id}/activate` | Model registry (one active model per task) |
| `POST /api/logs` | Interaction events for the manual-vs-assisted effort study |
| `GET /api/ai/status` | Which AI models are loaded |
| `POST /api/images/{id}/suggest` | Detector boxes, mapped to the project's classes |
| `POST /api/images/{id}/segment/prepare`, `POST /api/images/{id}/segment` | MobileSAM: encode the image ahead of time; clicks → polygon |

## Dataset

[Safety Helmet Detection](https://www.kaggle.com/datasets/andrewmvd/hard-hat-detection) (CC0), originally the
[Make ML Hard Hat Workers dataset](https://makeml.app/datasets/hard-hat-workers). Prepare the splits with:

```bash
cd backend
uv run python scripts/prepare_dataset.py --src D:/datasets/hard-hat-detection --out D:/datasets/hardhat-prepared
```

## Model training

The detector is fine-tuned on Google Colab (free T4 GPU) with
[eval/colab_finetune.ipynb](eval/colab_finetune.ipynb), tracked with MLflow and exported to
OpenVINO. Step-by-step guide: [docs/training.md](docs/training.md).

## Project documents

- [Project plan](docs/PROJECT_PLAN.md)
- [Decision records](docs/decisions/)
- [Learning log](docs/learning-log.md)
