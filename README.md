# AI-Assisted Image Annotation Platform

A web application for labelling images for object detection, where computer-vision models do
most of the work: a **fine-tuned YOLO11 detector** pre-labels every image, a **smart click**
turns one click into a correctly classed box, and **MobileSAM** outlines objects as polygons.
The annotator reviews and corrects, then exports a standard **COCO** or **YOLO** dataset.

Use case: labelling **construction-site helmet compliance**: every `helmet` (head wearing a hard
hat) and every `head` without one. Built as the individual project of the *AI Vision* module,
SRH Hochschule Heidelberg.

## Results at a glance

| | Result |
|---|---|
| Detector (YOLO11n, fine-tuned, 600 held-out test images) | **mAP50 0.945 · mAP50-95 0.632** |
| Same model exported to OpenVINO, on a laptop CPU | mAP50 0.943 (no accuracy lost) · **61 ms** per image |
| Smart click vs. SAM click (300 test objects, box IoU) | **0.77** vs. 0.47 mean IoU · 92 % vs. 46 % usable (IoU ≥ 0.5) |
| Effort study (60 test images, manual vs. assisted) | **−80 % time per image** (29.7 s → 5.8 s, Mann-Whitney p < 0.0001, r = 0.88) |
| Label quality vs. ground truth (F1, IoU ≥ 0.5) | manual 0.70 · assisted 0.93 · caveat: automation bias, see below |

Details: [eval/results/](eval/results/) (all numbers are reproducible, see *Evaluation*).

![Effort study: time per image, time per object, interactions](eval/results/figures/effort_boxplots.png)

## Quick start

Requirements: **Docker Desktop** (or Docker Engine with Compose v2), ~6 GB free disk, internet on
the first start.

```bash
git clone https://github.com/pramod05edu-afk/AI-annotation-tool.git
cd AI-annotation-tool
docker compose up --build
```

Open **http://localhost:3000**. The first start takes a few minutes (building the images,
downloading the models); the header shows *Backend ok* and the AI buttons enable themselves when
the models are ready.

| Service | URL |
|---|---|
| Web app | http://localhost:3000 |
| API docs (Swagger) | http://localhost:3000/api/docs |
| SeaweedFS filer UI (stored images) | http://localhost:8888 |

Default local-development credentials are in [.env.example](.env.example); copy it to `.env` to
change them. The database is exposed on host port **5433** (not 5432) to avoid clashing with a
locally installed PostgreSQL.

**Models are downloaded automatically** into `weights/` (git-ignored) on the first start:
the fine-tuned detector from this repository's
[GitHub Release `model-v1`](https://github.com/pramod05edu-afk/AI-annotation-tool/releases/tag/model-v1)
(SHA-256 verified), MobileSAM from the Ultralytics releases.

## Features

**Projects and images**
- Projects with their own label classes; drag-and-drop upload (PNG / JPEG / WEBP, validated by
  content), thumbnail grid with status (*to do / in progress / done*) and progress bar.

**Annotator**
- Boxes and polygons on a zoomable, pannable canvas; select, move, resize, edit polygon points.
- Keyboard-first: `B` box, `P` polygon, `V` select, `1`–`9` class, `Del`, `Ctrl+Z` undo,
  `A`/`D` previous/next image, `Enter` done. Autosave after every change.

**AI assistance** (*Assisted* mode)
- ✨ **Suggest** (`S`, automatic on new images): the fine-tuned detector pre-labels the image;
  confidence slider; AI boxes are dashed and keep their score.
- 🎯 **Smart click** (`C`): click an object → the detector's best box under the click, with its
  class, including weak candidates the automatic suggestions hid; already labelled objects are
  skipped. Detections are cached per image (repeat clicks ≈ 1 ms).
- 🪄 **Magic click** (`M`): MobileSAM outlines an object as a polygon (click = object,
  Shift / right click = not object); the image is encoded in the background when it opens.
- Every label records its **source** (`manual`, `ai`, `ai_edited`), score and **model version**.

**Data and evaluation**
- **Export** as COCO (boxes + polygons) or YOLO (txt + `data.yaml`), all or only *done* images,
  as a zip with images. Validated with `pycocotools` and Ultralytics.
- **Interaction log** of every action (time, mode, event) for the effort study; projects can
  **enforce** an annotation mode so a study cannot be run in the wrong mode.
- **Model registry**: every loaded model is registered with its test metrics; one active model
  per task.

## Architecture

```
Browser (React + Konva)
   │
   ▼
frontend: nginx ──/api──► backend: FastAPI (Python 3.11)
                            ├──► PostgreSQL 16 + pgvector   projects, images, annotations,
                            │                               model registry, interaction log
                            ├──► SeaweedFS (S3 API)         image files
                            └──► models, loaded once in the background, run on CPU
                                   ├─ YOLO11n fine-tuned  → OpenVINO  (suggest, smart click)
                                   └─ MobileSAM                       (magic click)
```

Key design decisions (justified in [the report](docs/) and [decision records](docs/decisions/)):

| Choice | Why |
|---|---|
| Fine-tune YOLO11n (Colab T4) instead of training from scratch | 3,800 images are too few for scratch training; YOLO11s was not more accurate on the test set (0.629 vs. 0.632 mAP50-95) but 3.6× larger |
| OpenVINO export | Intel CPU inference without a GPU; same accuracy (0.943 vs. 0.945 mAP50) |
| Smart click in addition to SAM | in this dataset a *helmet* box covers the whole head; SAM outlines only the visible hard hat (IoU 0.47); the detector knows the label definition (IoU 0.77) |
| Split by near-duplicate groups (dHash) | prevents the same scene from appearing in train and test (data leakage) |
| Typed API client generated from OpenAPI | the frontend cannot silently drift from the API; a test fails if it does |
| SeaweedFS (S3 API) for files, PostgreSQL for metadata | standard interfaces; storage is swappable for any S3 service |

## Evaluation

All scripts are in [backend/scripts/](backend/scripts/); results and figures in
[eval/results/](eval/results/).

| What | Script | Result file |
|---|---|---|
| Detector: mAP on CPU, precision/recall at the app threshold, recall by object size, failure cases, CPU speed | `evaluate_detector.py` | `detector_eval.json`, `failures/` |
| Click tools: SAM 1 click, SAM 2 clicks, smart click | `evaluate_click_tools.py` | `click_tools_eval.json` |
| Effort study: time, interactions, label quality, statistics | `analyze_effort.py` | `effort_study.{json,md}`, `figures/effort_boxplots.png` |
| Training curves, confusion matrix, PR / F1 curves (Colab) | `eval/colab_finetune.ipynb` | `figures/`, `detector_metrics.json` |

**Reproduce the effort-study analysis without the database or the dataset** (all raw events,
final labels and ground truth are in `eval/results/effort_study_raw.json`):

```bash
cd backend
uv sync
uv run python scripts/analyze_effort.py --raw ../eval/results/effort_study_raw.json
```

The detector and click-tool evaluations need the prepared dataset (see *Dataset*) and run on CPU
in a few minutes each, e.g. `uv run python scripts/evaluate_detector.py --data <prepared dataset>`.

**Effort study design** ([protocol](docs/effort-study.md)): 60 test images (3–15 objects each,
never used in training), two difficulty-balanced sets of 30 (184 vs. 183 objects), manual vs.
assisted, labelled in alternating blocks of 10; Mann-Whitney U test with rank-biserial effect size.

**Findings and limitations**
- Assisted labelling was **about 5× faster** and produced tighter boxes, especially on small
  objects (recall on objects < 32×32 px: 0.92 vs. 0.50 manual).
- **Automation bias:** 190 of 192 assisted labels were accepted unchanged; the AI's errors passed
  through. Assisted quality is therefore bounded by the model; a review step that highlights
  low-confidence boxes is the main improvement.
- One annotator (the developer); fast pace; quality measured against public labels that contain
  some missed heads. Small, distant heads are the detector's weakest case (recall 0.88 vs. 0.97).

## Dataset

[Safety Helmet Detection](https://www.kaggle.com/datasets/andrewmvd/hard-hat-detection)
(5,000 images, CC0), originally the
[Make ML Hard Hat Workers dataset](https://makeml.app/datasets/hard-hat-workers). Classes
`helmet` and `head` (`person` dropped: labelled in only 158 images). Splits by near-duplicate
groups: train 3,800 · val 500 · test 600 · export 100 ([eval/splits.json](eval/splits.json)).

```bash
cd backend
uv run python scripts/prepare_dataset.py --src <downloaded dataset> --out <prepared dataset>
```

**Delivered dataset:** the 100 *export* images labelled in the app (assisted, reviewed), exported
as COCO and YOLO: 100 images, 498 boxes (attached to the submission).

## Model training

Fine-tuned on Google Colab (free T4) with [eval/colab_finetune.ipynb](eval/colab_finetune.ipynb):
COCO-pretrained YOLO11n and YOLO11s, 60 epochs, 640 px, early stopping, MLflow tracking, test-set
evaluation and OpenVINO export. Guide: [docs/training.md](docs/training.md).

## Development

```bash
# Backend (Python 3.11, uv)
docker compose up -d postgres storage   # API tests use a real PostgreSQL (host port 5433)
cd backend
uv sync
uv run pytest                            # 112 tests; creates and migrates annotator_test
uv run ruff check . && uv run ruff format --check . && uv run mypy app
uv run alembic revision --autogenerate -m "describe change"   # after changing app/models.py

# Frontend (Node 24)
cd frontend
npm ci
npm run dev          # http://localhost:5173, proxies /api to localhost:8000
npm run lint && npm run build
npm run gen:api      # after an API change: regenerate the typed client
```

Git hooks (ruff, mypy, eslint, prettier, large-file and secret checks): `uvx pre-commit install`.
CI (GitHub Actions) runs lint, types, tests against PostgreSQL, the frontend build and the Docker
build on every pull request.

## API overview

Interactive docs: http://localhost:3000/api/docs

| Method & path | Purpose |
|---|---|
| `GET /api/health` | Database and storage status |
| `POST/GET /api/projects`, `GET/PATCH/DELETE /api/projects/{id}` | Projects, label classes, optional fixed annotation mode |
| `POST /api/projects/{id}/images` | Upload images (validated by content: PNG/JPEG/WEBP, size limit) |
| `GET /api/projects/{id}/images?status=` | List images with annotation counts |
| `GET/PATCH/DELETE /api/images/{id}`, `GET /api/images/{id}/file` | Image details, status, file stream |
| `GET/PUT /api/images/{id}/annotations` | Load / replace all annotations of an image (boxes, polygons) |
| `GET /api/projects/{id}/export?format=coco\|yolo&scope=all\|done` | Download the dataset as a zip with images |
| `GET /api/ai/status` | Which AI models are loaded |
| `POST /api/images/{id}/suggest` | Detector boxes, mapped to the project's classes |
| `POST /api/images/{id}/click-box` | Smart click: the detector's best box under a clicked point |
| `POST /api/images/{id}/segment/prepare`, `POST /api/images/{id}/segment` | MobileSAM: encode ahead of time; clicks → polygon |
| `GET/POST /api/models`, `POST /api/models/{id}/activate` | Model registry (one active model per task) |
| `POST /api/logs` | Interaction events for the effort study |

## Licences and credits

- Dataset: Safety Helmet Detection, CC0 (Kaggle `andrewmvd`, from Make ML).
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) (AGPL-3.0): detector training
  and inference. Fine for this open-source university project; commercial use would need an
  Ultralytics licence or an Apache-licensed detector.
- [MobileSAM](https://github.com/ChaoningZhang/MobileSAM) (Apache-2.0) via Ultralytics;
  [Segment Anything](https://segment-anything.com/) (Kirillov et al., 2023).
- [OpenVINO](https://github.com/openvinotoolkit/openvino) (Apache-2.0), FastAPI, React, Konva,
  PostgreSQL, SeaweedFS.
