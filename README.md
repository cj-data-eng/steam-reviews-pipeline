# Steam Reviews Pipeline

An end-to-end data pipeline that ingests player reviews from the Steam reviews API, lands them untouched in S3-style object storage, cleans and deduplicates them with Spark, and enriches them with sentiment and topic features. Everything runs locally from one `docker compose` stack and is driven from the command line.

Built as a portfolio project for consumer-data engineering: API ingestion, a medallion layout (raw / silver / gold), orchestration, data quality handling, privacy-aware processing, and a measured evaluation of the enrichment model.

## Architecture

```
Steam appreviews API
        |  (cursor pagination, requests)
        v
Airflow DAG: steam_reviews_extract
   1. extract_reviews  -> raw JSON pages
   2. spark_silver     -> clean, dedupe, pseudonymise, quarantine
   3. spark_gold       -> sentiment + topic features, daily aggregates
        |
        v
S3-compatible storage (Floci emulator, bucket steam-reviews-raw)
   raw/         immutable API responses
   silver/      cleaned Parquet, partitioned by review_date
   quarantine/  rejected rows with a reason
   gold/        enriched reviews + daily theme summary
```

| Component | Choice |
|---|---|
| Orchestration | Apache Airflow 2.10.5 (LocalExecutor, Postgres metadata DB) |
| Processing | PySpark 3.5.3, local mode, run inside the Airflow image |
| Storage | S3 API via the Floci AWS emulator (swap `S3_ENDPOINT_URL` for real AWS) |
| Enrichment | VADER sentiment + keyword topic tagging |
| Packaging | Docker Compose, custom Airflow image (Java 17 + PySpark) |

## Data layers

| Layer | Path | What it holds |
|---|---|---|
| Raw | `raw/app_id=<id>/ingest_date=<date>/run=<time>/page_NNNN.json` | Unmodified API responses, one file per page. A retry creates a new `run=` folder and never overwrites. |
| Silver | `silver/reviews/app_id=<id>/review_date=<date>/` | One row per review, typed and flattened, deduplicated on `recommendation_id` (latest `updated_at` wins). |
| Quarantine | `quarantine/reviews/app_id=<id>/` | Rows that failed a quality rule, each with a `reject_reason`. |
| Gold | `gold/reviews_enriched/`, `gold/daily_theme_sentiment/` | Per-review sentiment score, label, topics and model version; plus a daily aggregate by topic. |

## Design decisions

- **Raw is never modified.** Every cleaning decision lives in code, so any layer can be rebuilt from raw.
- **Idempotent by design.** Silver re-reads all raw runs, deduplicates, and overwrites only the date partitions it touches. Running the pipeline repeatedly does not create duplicates (see results).
- **Reviews can be edited**, so deduplication keeps the version with the latest `updated_at`.
- **Quarantine, not silent drops.** Rows with a missing ID, missing timestamp or empty text are written to quarantine with a reason.
- **Pseudonymisation.** The reviewer's Steam ID is replaced by a SHA-256 hash, and the display name, profile URL and avatar are dropped from silver. This is pseudonymisation, not anonymisation: an unsalted hash of a public identifier can be reversed by guessing IDs. A production version would use a keyed hash (HMAC) with a secret held outside the repo. The raw layer still contains the original fields and would need access controls and a retention policy.
- **Least privilege.** Containers run as a non-root user (`AIRFLOW_UID`), the Airflow UI is bound to `127.0.0.1`, and no Docker socket is mounted. The `admin/admin` login is for local use only.
- **Floci instead of LocalStack.** LocalStack's current images require an account token; Floci is open source and needs none. S3A change detection is disabled in the Spark jobs for the emulator only and should stay at its default on real AWS.
- **Spark runs inside the Airflow container.** This keeps the project to a single `docker compose up` and avoids mounting the Docker socket. The trade-off is that orchestration and compute share a process; at larger scale the Spark step would submit to a separate cluster.

## Results

Run on the most recent reviews for one game (Steam app 730), 11-16 October 2026.

**Pipeline behaviour**

| Metric | Value |
|---|---|
| Raw reviews read (several overlapping runs) | 6,600 |
| After deduplication | 3,000 |
| Silver rows | 2,998 |
| Quarantined | 2 (empty review text) |

Re-running the DAG on overlapping data collapses back to the same silver row count, which is the idempotency check.

**Sentiment scoring.** The reviews carry a free label (`voted_up`, whether the reviewer recommends the game). I compared two scorers on a time-based split: trained on the oldest 80% of reviews and tested on the newest 600 (15-16 October), so the model never saw what it was graded on. Full output is in `docs/model_comparison.txt`.

| Scorer | AUC on held-out reviews |
|---|---|
| VADER (general-purpose, rule-based) | 0.658 |
| TF-IDF + logistic regression trained on `voted_up` | 0.888 |

Accuracy was 0.803 for the trained model against 0.767 for always predicting "recommended", so AUC is the more informative measure on this imbalanced label. The terms the trained model learned include gaming vocabulary that VADER's dictionary lacks (`cheaters`, `anticheat`, `gg`, `unplayable`).

**Limitations.** The label is a proxy for sentiment (people recommend games they complain about). The sample is one game over five days, the share of recommendations moved from 84% in training to 77% in the test window (some drift), and the trained model is currently an offline experiment rather than a pipeline stage. The gold layer uses VADER.

## Running it

Prerequisites: Docker with Compose, the AWS CLI, and Python 3 (only needed for running the extractor outside Airflow).

```bash
echo "AIRFLOW_UID=$(id -u)" > .env
mkdir -p logs plugins
docker compose up -d --build
export AWS_ACCESS_KEY_ID=test AWS_SECRET_ACCESS_KEY=test AWS_DEFAULT_REGION=eu-west-2
aws --endpoint-url http://localhost:4566 s3 mb s3://steam-reviews-raw
docker compose exec airflow-scheduler airflow dags test steam_reviews_extract
```

The credentials above are dummy values for the local emulator. The bucket must be recreated whenever the Floci container is recreated, because the emulator does not persist data. Use `-c '{"app_id": 730, "max_pages": 30}'` on the last command to pull more reviews, and the Airflow UI is at http://localhost:8080.

## Repository layout

```
dags/steam_reviews_extract.py   Airflow DAG: extract -> silver -> gold
src/steam_reviews/extract.py    API client and raw landing
spark_jobs/silver_reviews.py    clean, dedupe, pseudonymise, quarantine
spark_jobs/gold_reviews.py      sentiment + topics + aggregates
experiments/compare_scorers.py  VADER vs TF-IDF + logistic regression
docs/model_comparison.txt       saved output of the comparison
Dockerfile                      Airflow image with Java 17 and PySpark
docker-compose.yml              Airflow, Postgres, Floci, optional Spark service
```

## Roadmap

- Incremental extraction using a stored high-water mark instead of re-fetching the latest pages
- Train and version the TF-IDF model inside the pipeline and score with it in gold
- Unit tests for transformation logic and a GitHub Actions workflow
- Serving layer and a dashboard of sentiment by topic over time
- Compare a pretrained transformer against the current scorers on the same holdout