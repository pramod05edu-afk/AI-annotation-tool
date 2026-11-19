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
cd backend
uv sync
uv run pytest
uv run ruff check . && uv run ruff format --check . && uv run mypy app

# Frontend (Node)
cd frontend
npm ci
npm run dev        # http://localhost:5173, proxies /api to localhost:8000
npm run lint && npm run build
```

Install the git hooks once: `uvx pre-commit install`.

## Dataset

[Safety Helmet Detection](https://www.kaggle.com/datasets/andrewmvd/hard-hat-detection) (CC0), originally the
[Make ML Hard Hat Workers dataset](https://makeml.app/datasets/hard-hat-workers). Prepare the splits with:

```bash
cd backend
uv run python scripts/prepare_dataset.py --src D:/datasets/hard-hat-detection --out D:/datasets/hardhat-prepared
```

## Project documents

- [Project plan](docs/PROJECT_PLAN.md)
- [Decision records](docs/decisions/)
- [Learning log](docs/learning-log.md)
