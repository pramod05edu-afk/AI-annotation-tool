# Learning log

One entry per day: what I learned, problems I hit, how I solved them.
These become interview stories ("Tell me about a hard bug you fixed").

## 2026-11-18
- **Learned:**
- **Problem → solution:** Docker said `failed to connect to the docker API ... dockerDesktopLinuxEngine` → WSL was not installed and Docker Desktop was not running → installed WSL (`wsl --install --no-distribution`), restarted, started Docker Desktop.
- **Problem → solution:** Two Kaggle car datasets looked popular but had a No-Derivatives license / no stated license → learned to check license, source and video-frame duplicates first.
- **Dataset finding:** `person` class labeled in only 158 of 5,000 images → dropped it; near-duplicate images found → split by similarity group to avoid train/test leakage.
- **Problem → solution:** `minio/minio` image no longer exists on Docker Hub (quay.io needs login) → switched to SeaweedFS; only `docker-compose.yml` changed because the code uses the standard S3 API (decision 0002). Lesson: depend on standard interfaces, pin versions.
- **Problem → solution:** Docker marked the backend *unhealthy* while storage was down → `/api/health` took 10.4 s because boto3 retried with backoff → gave the health check its own client with 1 s timeout and no retries. Lesson: health checks must fail fast.
