# AI-Assisted Image Annotation Platform — Project Plan

**Course:** AI Vision (SRH Hochschule Heidelberg)
**Deadline:** Sunday, 30 Nov 2026, 23:59 (Moodle)
**Hardware:** Intel i5-1335U laptop, 16 GB RAM, no NVIDIA GPU → app runs on CPU; Google Colab (T4) for GPU work.
**Goals:** (1) meet every course requirement for a strong grade, (2) build a portfolio project at industry standard, (3) learn the stack well enough to defend it in interviews.
**Repo:** https://github.com/pramod05edu-afk/AI-annotation-tool (cloned to `D:\Agents\AI-annotation-tool`)
**Dataset (decided 18 Nov):** Safety Helmet Detection — https://www.kaggle.com/datasets/andrewmvd/hard-hat-detection — CC0 public domain, 5,000 images, Pascal VOC boxes, classes helmet / head / person; original source: Make ML "Hard Hat Workers" (https://makeml.app/datasets/hard-hat-workers). Target users: safety teams building helmet-compliance datasets.

### Revised schedule (18 Nov — supersedes the dates in Section 6)
Phase 0 took until 18 Nov. Dataset findings: drop the sparse `person` class (2 classes: helmet, head); small objects (zoom/pan needed); near-duplicate images (split by similarity group); 5,000 images → train ~3,700 / val ~500 / test ~600 / export ~100.

| Date | Work |
|---|---|
| 18–19 Nov | Phase 1 (skeleton, Docker, tooling, CI) + dataset preparation + exercises E1–E4 |
| 20–21 Nov | Phase 2 (backend) · **start Colab fine-tuning (Phase 5) — data is ready, training runs in parallel** |
| 22–24 Nov | Phase 3 (annotation UI, incl. zoom/pan) |
| 25–26 Nov | Phase 4 (AI integration) + Phase 6 (export, logging) |
| 27 Nov | Phase 7 evaluation: scripts + effort study |
| 28 Nov | Phase 7 finish + Phase 9 (final dataset, clean install, README) |
| 29 Nov | Phase 10 (report, video) |
| 30 Nov | Buffer, submit before 23:59 |

Phase 8 (CLIP) only if ahead of this schedule.

---

## 1. Requirements checklist (from the course slides)

- [ ] Users can upload images and label them
- [ ] At least one AI model assists *inside the app* (not a separate script)
- [ ] Annotations are saved and exportable in a standard format
- [ ] Evidence: model quality (mAP / IoU) + effort saved (manual vs. assisted) + failure cases
- [ ] Every feature / model / tech choice justified in the report
- [ ] All models, libraries, datasets cited; no forking/wrapping existing annotation tools (CVAT, Label Studio, …)

**Deliverables**
1. Report (Moodle)
2. Git repo — README, runs with `docker compose up`, includes evaluation scripts
3. Demo video (Moodle) — upload → AI-assisted annotation → correction → export
4. Exported annotated dataset (link) — standard format + images

**Before starting:** check the course rules on using AI assistants (allowed? must it be disclosed?). Ask Prof. Vu if unclear, and disclose in the report if required.

---

## 2. Tech stack

| Layer | Choice | Justification (for the report) |
|---|---|---|
| Frontend | React + Vite + TypeScript (strict), **react-konva** | Component UI; Konva gives drawing, dragging and resizing of shapes on a canvas |
| API client | TypeScript types generated from FastAPI's OpenAPI schema | Frontend and backend stay in sync automatically |
| Web server | nginx | Serves the built frontend and forwards `/api` to the backend; one URL for the user |
| Backend | Python 3.11 + **FastAPI** + SQLAlchemy 2 + **Alembic** + pydantic-settings | Same language as the models; async; auto API docs; versioned DB migrations; config from env vars |
| Database | **PostgreSQL 16 + pgvector** | Standard production database, safe for multiple users, JSONB for annotation geometry, vector search in the same database |
| Image storage | **SeaweedFS** (S3-compatible; replaced SeaweedFS — decision 0002) | Images kept out of the DB; same `boto3` code works on AWS S3 |
| Box model | **YOLO11n/s** (Ultralytics), exported to **OpenVINO** | Real-time on Intel CPU; fine-tunable on Colab |
| Mask model | **MobileSAM** (fallback: SAM 2.1 tiny) | One click gives a mask; fast enough on CPU |
| Embeddings *(stretch)* | **OpenCLIP ViT-B/32** → pgvector | Similar-image search, duplicate detection |
| Experiment tracking | **MLflow** (local) or Weights & Biases (free tier) | Logs fine-tuning runs: parameters, metrics, weights |
| GPU work | Google Colab (T4) + Google Drive | Fine-tuning and large evaluation runs |
| Export | COCO JSON, YOLO txt | The two most widely used formats |
| Packaging | Docker Compose (4 services), multi-stage builds, non-root user | One-command startup; production-style images |
| Code quality | **uv** (deps + lockfile), **ruff**, **mypy**, ESLint, Prettier, **pre-commit** | Consistent, checked code on every commit |
| Testing | **pytest** (+ optional Playwright end-to-end test) | Correctness of export, API and model wrappers |
| CI | **GitHub Actions** | Lint, type-check, test and build on every push; status badge in README |

### Architecture

```
Browser ──► frontend (nginx :3000) ──/api──► backend (FastAPI :8000)
                                               ├──► postgres (pgvector)  : projects, images, annotations,
                                               │                           model versions, logs, embeddings
                                               ├──► storage (S3, SeaweedFS): image files, export zips
                                               └──► models (in-process, thread pool)
                                                     YOLO11 (OpenVINO), MobileSAM, [CLIP]
```

### Not in scope (mention under "Limitations / future work")
Kubernetes, separate model microservices, full login system, message queues (Kafka), DVC, extra models (e.g. Grounding DINO).

---

## 3. Repository layout

```
ai-annotation-tool/
├─ backend/
│  ├─ app/
│  │  ├─ main.py            # FastAPI app + lifespan (load models, create bucket)
│  │  ├─ config.py          # pydantic-settings, reads env vars
│  │  ├─ db.py              # SQLAlchemy engine/session
│  │  ├─ models.py          # ORM tables
│  │  ├─ schemas.py         # Pydantic request/response models
│  │  ├─ storage.py         # S3 / boto3 helpers
│  │  ├─ routers/           # projects, images, annotations, ai, export, logs, models
│  │  ├─ services/          # export (COCO/YOLO), geometry (box conversions, IoU)
│  │  └─ ml/                # base.py (Detector/Segmenter interfaces), yolo.py, sam.py, clip.py, weights.py
│  ├─ tests/                # pytest
│  ├─ alembic/              # migrations
│  ├─ pyproject.toml + uv.lock
│  └─ Dockerfile            # multi-stage, non-root
├─ frontend/                # Vite React app + nginx.conf + Dockerfile
├─ eval/                    # eval scripts, Colab notebooks, results/
├─ scripts/                 # seed_demo.py (loads a sample project)
├─ docs/
│  ├─ decisions/            # one short decision record per choice (→ report)
│  └─ learning-log.md       # daily notes, problems solved (→ interview stories)
├─ .github/workflows/ci.yml
├─ .pre-commit-config.yaml
├─ docker-compose.yml
├─ .env.example             # local-only default credentials (copied to .env)
└─ README.md
```

---

## 4. How we work

### Division of work
| Area | Who leads |
|---|---|
| ML code (YOLO/SAM wrappers, IoU, mAP, effort analysis, export) | **You write it**; Claude guides and reviews |
| Backend (FastAPI, SQLAlchemy, Postgres) | **Together**: Claude scaffolds, you write endpoints step by step |
| Docker, CI, tooling | **Claude sets it up**, explains every line; you make small changes |
| Frontend (React / Konva) | **Claude builds most of it** with explanations; you make small changes |

### Working rules
1. **Explain first, then build** — every phase starts with what we build and why.
2. **You type the important parts** — evaluation code, model wrappers, export.
3. **Explain it back** — after each phase, explain how it works in your own words; Claude asks interview-style follow-up questions.
4. **Golden rule:** never commit code you can't explain. Ask "explain this file line by line" whenever unsure.
5. **Write a decision record** (`docs/decisions/NNN-title.md`: context → options → choice → why) for every important choice.
6. **Update `docs/learning-log.md` daily** — what you learned, problems hit, how you solved them.

### How deeply to learn each tool
| Level | Tools | Goal |
|---|---|---|
| 🔴 Deep | YOLO (one-pass detection, NMS), SAM (encoder / prompt encoder / decoder), IoU, precision/recall, mAP, fine-tuning & transfer learning, CLIP & embeddings, Python, NumPy, FastAPI, SQL basics | Explain on a whiteboard |
| 🟡 Working | Docker / Compose, PostgreSQL + pgvector, Git, pytest, CI, MLflow | Use it and justify the choice |
| 🟢 Basic | React + Konva, TypeScript, nginx, SeaweedFS | Read it and change it with help |

---

## 5. Python warm-up (alongside Phases 0–2)

**Topics to close the gap from "Python basics":**
| Topic | Needed for | Resource |
|---|---|---|
| Type hints | FastAPI, Pydantic | FastAPI docs — "Python Types Intro" |
| Classes, `@dataclass` | Model wrappers, schemas | Python tutorial, ch. 9 |
| Virtual environments, `uv` | Reproducible dependencies | uv docs — "Getting started" |
| Decorators, context managers | Routes (`@app.get`), DB sessions (`with`) | Real Python articles |
| async / await basics | FastAPI, non-blocking inference | FastAPI docs — "Concurrency and async / await" |
| **NumPy arrays** (shape, dtype, slicing, H×W×3) | Every image is a NumPy array | NumPy — "absolute basics for beginners" |
| OpenCV basics (read, resize, draw, contours) | Image loading, mask → polygon | OpenCV-Python tutorials |

**Exercises** (all reused in the project; Claude reviews each one):
- [ ] E1 — Load an image with OpenCV; print `shape`/`dtype`; BGR→RGB; grayscale
- [ ] E2 — Draw ground-truth boxes from your dataset with `cv2.rectangle`
- [ ] E3 — Convert COCO `[x, y, w, h]` ↔ YOLO normalized `[cx, cy, w, h]` (→ `services/geometry.py`)
- [ ] E4 — `iou(box_a, box_b)` with NumPy (→ evaluation; classic interview question)
- [ ] E5 — Binary mask → polygon with `cv2.findContours` + `cv2.approxPolyDP` (→ SAM integration)
- [ ] E6 — pytest tests for E3 and E4 (→ `backend/tests/`)

---

## 6. Phase-by-phase plan

Each day: ~1–2 h **learn** (morning) → **build** (afternoon) → learning log (evening). Tests are written inside each phase, not at the end.

### Phase 0 — Setup & dataset (Sun 15 Nov)
**Learn:** Git basics; Docker official "Get started" guide.
1. Install Git, Docker Desktop (WSL2), Node.js LTS, Python 3.11, uv, VS Code.
2. Create `C:\Users\<you>\.wslconfig`:
   ```
   [wsl2]
   memory=8GB
   processors=6
   ```
3. Create the GitHub repo.
4. Pick a dataset **with ground truth** (~300 images, 3–5 classes): COCO val2017 subset, or a domain dataset from Kaggle / Roboflow Universe. Write decision record 001 (target users + dataset).
5. Split into **non-overlapping** sets:
   - **Train/val (~150):** fine-tuning on Colab only
   - **Test (~100):** mAP/IoU evaluation and the timing study (never trained on)
   - **Final export (~50–100):** annotated with your tool → deliverable #4
6. Evening: exercises E1–E2.

✅ Done when: `docker run hello-world` works; dataset downloaded and split; repo cloned; E1–E2 done.

### Phase 1 — Skeleton, Docker Compose, tooling, CI (Mon 16 Nov)
**Learn:** Docker Compose docs; YAML basics; type hints.
1. Compose services:
   - `postgres`: `pgvector/pgvector:pg16`, named volume, healthcheck (`pg_isready`)
   - `storage`: `chrislusf/seaweedfs` (S3 gateway on :8333, filer UI on :8888), named volume, healthcheck
   - `backend`: waits for postgres + storage to be **healthy**; runs `alembic upgrade head`, then `uvicorn`
   - `frontend`: Vite build served by nginx; forwards `/api` to the backend
2. Backend: `uv` project with lockfile; **CPU-only PyTorch** (`--index-url https://download.pytorch.org/whl/cpu`); multi-stage Dockerfile, non-root user, `.dockerignore`, weights on their own volume.
3. Config via pydantic-settings; `.env.example` with local-only defaults (compose falls back to defaults so `docker compose up` works out of the box).
4. Backend creates the SeaweedFS bucket on startup if missing.
5. Tooling: ruff, mypy, ESLint, Prettier, TypeScript strict, pre-commit hooks.
6. CI (`.github/workflows/ci.yml`): backend lint + type-check + pytest (with a Postgres service); frontend lint + type-check + build; `docker compose build`.
7. Evening: exercises E3–E4.

✅ Done when: `docker compose up` → `http://localhost:3000` shows the React page; `GET /api/health` reports DB + storage OK; SeaweedFS filer UI at `:8888`; CI is green on GitHub.

### Phase 2 — Backend: data model, storage, API (Tue 17 – Wed 18 Nov)
**Learn:** FastAPI official tutorial (first ~10 sections); SQLBolt; SQLAlchemy 2.0 ORM quickstart; Alembic basics; S3 object storage concept.

**Tables** (Alembic; `CREATE EXTENSION vector` in the first migration):
- `projects`: id, name, classes (JSONB), created_at
- `images`: id, project_id, storage_key, filename, width, height, status (todo / in_progress / done), embedding `vector(512)` NULL *(stretch)*
- `model_versions`: id, name (e.g. `yolo11s-coco`, `yolo11s-ft-v1`), task (detect / segment / embed), weights_uri, metrics (JSONB), is_active, created_at
- `annotations`: id, image_id, class_name, type (box / polygon), geometry (JSONB, **original image pixels**), source (manual / ai / ai_edited), score, model_version_id NULL
- `interaction_logs`: id, image_id, session_id, mode (manual / assisted), event, payload (JSONB), ts

**Storage:** uploads go to SeaweedFS under `projects/{id}/images/{uuid}.{ext}`; backend streams them via `/api/images/{id}/file`. Upload checks: allowed types, size limit, image must decode.

**Endpoints:**
```
GET    /api/health
POST   /api/projects                         GET /api/projects
POST   /api/projects/{id}/images  (multi-upload)
GET    /api/projects/{id}/images             GET /api/images/{id}/file
GET    /api/images/{id}/annotations          PUT /api/images/{id}/annotations   (save all)
PATCH  /api/images/{id}                      (status)
POST   /api/images/{id}/predict   {conf}           → YOLO boxes        (stub for now)
POST   /api/images/{id}/segment   {points, labels} → polygon           (stub for now)
GET    /api/models                           POST /api/models/{id}/activate
GET    /api/projects/{id}/export?format=coco|yolo  → zip
POST   /api/logs
```
**Tests:** API tests for projects, uploads, annotations against a test Postgres database.
**Evening:** exercises E5–E6.

✅ Done when: via `/docs` you can create a project, upload images (visible in SeaweedFS), save and reload annotations; tests pass in CI.

### Phase 3 — Annotation UI (Thu 19 – Sat 21 Nov)
**Learn:** react.dev "Learn" (components, state, effects); react-konva examples.
1. Generate the TypeScript API client from the OpenAPI schema.
2. **Projects page:** create a project, define classes.
3. **Project page:** drag-and-drop upload, thumbnail grid with status badges, progress bar, export button.
4. **Annotator page:**
   - Konva stage scaled to fit; convert screen ↔ image coordinates (`x_img = x_screen / scale`); save only image coordinates
   - Tools: Select (V), Box (B), Polygon (P), Delete (Del); Transformer for drag/resize
   - Class dropdown + number keys 1–9
   - Sidebar annotation list (hover highlights the shape)
   - Previous / next (A / D), autosave on image change, "Mark done" (Enter)
   - Undo (Ctrl+Z) — nice to have

✅ Done when: you can fully label 5 images by hand, reload, and everything is still there.

### Phase 4 — AI integration (Sun 22 – Mon 23 Nov)
**Learn:** Ultralytics docs (predict, train, export); SAM paper (abstract, Fig. 1, model section); CS231n detection & segmentation lecture.

**Backend (`app/ml/`):**
1. `Detector` / `Segmenter` interfaces in `base.py`; YOLO and MobileSAM implement them.
2. Load models once at startup (lifespan); download weights to the weights volume if missing; register them in `model_versions`.
3. Run inference in a thread pool with a concurrency limit; log latency per call.
4. **YOLO:** `predict(img, conf)` → map classes → `[{class, bbox:[x,y,w,h], score}]`. Export to OpenVINO; record latency before/after.
5. **MobileSAM:** image embedding cached per image (LRU); clicks → mask → polygon (E5) + bbox.
6. Tests: wrappers on a small fixture image (output shape/format, not exact values).

**Frontend:**
- **Manual / Assisted mode toggle** (manual hides AI tools)
- **✨ Suggest** + confidence slider → dashed suggestion boxes; accept / reject / edit; saved as `ai` or `ai_edited` with `model_version_id`
- **Magic-click (M):** left = positive point, right = negative point; live preview; Enter accepts
- Loading spinners and error messages

✅ Done when: Suggest (< ~1 s) and magic-click work inside Docker on the laptop.

### Phase 5 — Fine-tune on Colab + experiment tracking (in parallel, 22 – 23 Nov)
1. Upload the train/val split (YOLO format) to Google Drive.
2. `eval/colab_finetune.ipynb`:
   ```python
   from google.colab import drive; drive.mount('/content/drive')
   !pip install ultralytics mlflow
   !yolo train model=yolo11s.pt data=/content/drive/.../data.yaml epochs=50 imgsz=640
   !yolo val   model=runs/detect/train/weights/best.pt data=...   # test split
   ```
3. Log parameters, metrics and weights with MLflow (store the `mlruns/` folder on Drive) or W&B.
4. Publish `best.pt` as a GitHub Release asset; register it as a new row in `model_versions` and activate it.
5. Record mAP **before** (COCO-pretrained) and **after** fine-tuning.
6. Document the **human-in-the-loop loop**: annotate in app → export → fine-tune → register new version → measure again.

✅ Done when: both mAP numbers recorded and tracked; app runs with the fine-tuned version.

### Phase 6 — Export & interaction logging (Tue 24 Nov)
**Learn:** COCO data format page; YOLO label format.
1. **COCO export:** `images`, `annotations` (bbox `[x,y,w,h]`, segmentation, area, iscrowd=0), `categories` + images → zip.
2. **YOLO export:** `images/`, `labels/*.txt` (`cls cx cy w h`, normalized 0–1), `data.yaml`.
3. **Tests:** round-trip (export → re-import → coordinates match); validate COCO with `pycocotools`, YOLO with Ultralytics.
4. Frontend logs: `image_opened`, `annotation_added/edited/deleted`, `suggestion_accepted/rejected`, `click`, `image_done` — with mode and session id.
5. `scripts/seed_demo.py`: one command loads a sample project.

✅ Done when: export round-trip tests pass; events visible in `interaction_logs`.

### Phase 7 — Evaluation (Wed 25 – Thu 26 Nov)
**Learn:** IoU, precision/recall, mAP; mean ± std, Wilcoxon signed-rank test.

All scripts in `eval/`, one command each, output to `eval/results/`:
1. `eval_detection.py` — mAP@0.5 and mAP@0.5:0.95 per class on the **test** split, pretrained vs. fine-tuned.
2. `eval_sam.py` — simulated clicks at ground-truth object centers → mean mask IoU for 1 / 3 / 5 clicks.
3. **Effort study:**
   - 30 test images in **Manual** mode, 30 *different* test images in **Assisted** mode
   - Ideally 2–3 extra participants; **alternate which mode each starts with**
   - `eval_effort.py` (SQL on `interaction_logs`): time per image, clicks per object, **corrections per image**, % suggestions accepted unchanged, % objects added manually
   - Report **mean ± std, n**, and a paired Wilcoxon test; state the small sample honestly
   - Quality check: each mode's labels vs. ground truth (IoU / mAP) → faster *and* not worse
4. Failure cases: 6–10 screenshots (small/occluded objects, wrong class, SAM leaking, missed objects).
5. Charts: time per image (bar + error bars), precision-recall curve, SAM IoU vs. clicks, latency table (PyTorch vs. OpenVINO).

✅ Done when: all numbers and charts are reproducible from the scripts.

### Phase 8 — Stretch: CLIP similar images (Fri 27 Nov morning, max ~4 h)
**Only if Phases 1–7 are done.** **Learn:** CLIP overview; cosine similarity; HNSW indexes.
1. Compute OpenCLIP ViT-B/32 embeddings on upload → `images.embedding`; HNSW index (`vector_cosine_ops`).
2. **Similar images** panel: `ORDER BY embedding <=> :query LIMIT 8`.
3. **Duplicate warning** on upload (distance below a threshold).
4. Small check: precision@5 of similar-image search using ground-truth classes.

✅ Done when: works in Docker; otherwise list it under "future work".

### Phase 9 — Final dataset, clean install, README (Fri 27 Nov afternoon)
1. Annotate the final-export split in Assisted mode → export COCO + YOLO with images → upload the zip (Google Drive or GitHub Release) → share link.
2. Clean-install test: delete Docker images and volumes → fresh `git clone` → `docker compose up` → everything works (weights download, migrations, seed script).
3. **README (recruiter-ready):** demo GIF at the top, CI badge, one-paragraph pitch, architecture diagram, results table (mAP, IoU, time saved), quick start (`docker compose up`), URLs (app `:3000`, API docs, SeaweedFS filer UI), usage + keyboard shortcuts, how to run `eval/`, links to decision records, citations and licenses (Ultralytics AGPL-3.0; MobileSAM / SAM Apache-2.0; OpenCLIP MIT; PostgreSQL, pgvector, SeaweedFS).

### Phase 10 — Report & demo video (Sat 28 – Sun 29 Nov)
**Report** (built from the decision records and eval results):
1. Problem and target users
2. Features → the user problem each one solves
3. Architecture diagram + how the models are integrated (interfaces, model versions, thread pool)
4. Model choices (YOLO11, MobileSAM, [CLIP]) + alternatives considered, OpenVINO latency, Colab fine-tuning + experiment tracking
5. Evaluation: mAP, IoU, effort study with statistics, failure cases
6. Limitations (CPU latency, small user study, no auth, single-machine) + future work + references
7. AI-assistance disclosure if the course requires it

**Demo video (3–5 min, OBS):** create project → upload → Suggest → accept → fix a wrong box → magic-click mask → *(similar images)* → mark done → export → open the exported JSON.

### Mon 30 Nov — Buffer & submit before 23:59
Report + video on Moodle; links to the repo and exported dataset.

### After submission — portfolio polish
- Interview preparation (Section 8)
- Optional Playwright end-to-end test
- Optional live demo on Hugging Face Spaces (Docker, free CPU tier) → link on your CV

---

## 7. Fallback: what to cut if behind (in this order)
1. Phase 8 (CLIP)
2. SeaweedFS → local folder (keep Postgres)
3. UI extras: undo, extra shortcuts, polygon editing
4. MLflow → a results table in the notebook

**Never cut:** the AI features, export, evaluation, tests for export/IoU, README, report.

---

## 8. Interview questions to master

**Computer vision / ML**
- How does YOLO detect objects in one pass? What is NMS?
- What are IoU and mAP@0.5:0.95, and why use them?
- How does SAM turn a click into a mask? Why cache the image embedding?
- What changed when you fine-tuned? Why not train from scratch?
- What is CLIP, and how does similar-image search work?
- What were your failure cases, and why do they happen?

**Engineering**
- Walk me through the architecture. What happens when a user clicks "Suggest"?
- Why Postgres instead of MongoDB? Why images in S3 object storage, not the database?
- How did you keep inference from blocking the API?
- How do you track which model produced which annotation?
- How would you scale to 100 users? (separate inference service, GPU, job queue, caching)
- How do you know your export is correct? (round-trip tests)

**Project / behavioral**
- How did you prove the AI saves time? (effort study + statistics)
- Tell me about a hard bug you fixed. (learning log)
- What would you do differently? (limitations)

---

## 9. Key risks

| Risk | Mitigation |
|---|---|
| Learning load too high | Learn just-in-time; depth levels in Section 4; fallback cuts in Section 7 |
| Canvas UI takes longer than planned | Claude leads the frontend; cut UI extras first |
| Docker / startup problems found late | Healthchecks + CI builds images from Phase 1; clean-install test in Phase 9 |
| Evaluation left too late | Logging built in Phases 4/6; eval scripts in Phase 7 with buffer after |
| CPU inference too slow | YOLO11n + OpenVINO, MobileSAM, cached SAM embeddings, thread pool |
| Can't explain own code in interviews | Golden rule, explain-back after each phase, decision records, learning log |
