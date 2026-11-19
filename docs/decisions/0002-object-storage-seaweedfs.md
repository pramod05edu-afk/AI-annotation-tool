# 0002 — Object storage: SeaweedFS instead of MinIO

- **Date:** 2026-11-18
- **Status:** accepted

## Context
Uploaded images should not live in PostgreSQL (databases handle large binary files poorly) but in
S3-compatible object storage — the same API used by AWS S3 and most ML data pipelines. The plan
named MinIO. When starting the stack, `minio/minio` no longer existed on Docker Hub
("repository does not exist") and `quay.io/minio/minio` required a login (HTTP 401): MinIO stopped
distributing free community images. The app must start with one command (`docker compose up`) on
any machine, so a login-gated image is not acceptable.

## Options considered
1. **SeaweedFS** — mature (since 2014), Apache-2.0, actively maintained, S3-compatible gateway.
   Single container in all-in-one mode. No MinIO-style admin console (has a simple filer UI).
2. **RustFS** — MinIO look-alike including console, Apache-2.0, but a young project.
3. **Local folder (Docker volume)** — simplest, but drops the S3 API; moving to the cloud later
   would need code changes.

## Decision
SeaweedFS (`chrislusf/seaweedfs:4.54`, pinned), S3 API on port 8333, filer UI on port 8888.

## Why
- The backend talks plain S3 through `boto3`, so the swap changed only `docker-compose.yml` and
  default settings — **no application code changed**. This is exactly why we coded against the S3
  API rather than a vendor SDK.
- Verified before adopting: health endpoint (`/healthz`) ready in ~3 s; bucket create, upload and
  download work; **wrong secret → `SignatureDoesNotMatch`, unknown key → `InvalidAccessKeyId`**.
- Credentials come from environment variables (`AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`),
  so no secrets file is committed.

## Consequences
- Pinned image tag protects against surprise upstream changes (the MinIO lesson).
- No admin console for users/policies; not needed for a single-user local tool.
- In production the same code would point at AWS S3 or GCS by changing `S3_ENDPOINT_URL`.
- Lesson for the report/interviews: depend on **standard interfaces** (S3 API) and **pin versions**,
  so a disappearing dependency is a config change, not a rewrite.
