import argparse
import json
import os
import time
from datetime import datetime, timezone

import boto3
import requests

BASE_URL = "https://store.steampowered.com/appreviews/{app_id}"


def get_s3():
    # S3_ENDPOINT_URL points at Floci locally; unset it to use real AWS
    return boto3.client("s3", endpoint_url=os.environ.get("S3_ENDPOINT_URL"))


def fetch_page(app_id, cursor, num_per_page=100):
    params = {
        "json": 1,
        "num_per_page": num_per_page,
        "language": "english",
        "filter": "recent",
        "purchase_type": "all",
        "cursor": cursor,
    }
    resp = requests.get(BASE_URL.format(app_id=app_id), params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if data.get("success") != 1:
        raise RuntimeError(f"Steam API returned success={data.get('success')}")
    return data


def extract(app_id, max_pages=5, bucket="steam-reviews-raw", num_per_page=100, sleep_s=1.0):
    s3 = get_s3()
    run_ts = datetime.now(timezone.utc)
    prefix = f"raw/app_id={app_id}/ingest_date={run_ts:%Y-%m-%d}/run={run_ts:%H%M%S}"
    cursor = "*"
    pages = 0
    total = 0
    while pages < max_pages:
        data = fetch_page(app_id, cursor, num_per_page)
        reviews = data.get("reviews", [])
        if not reviews:
            break
        pages += 1
        total += len(reviews)
        key = f"{prefix}/page_{pages:04d}.json"
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=json.dumps(data).encode("utf-8"),
            ContentType="application/json",
        )
        print(f"wrote s3://{bucket}/{key} ({len(reviews)} reviews)")
        new_cursor = data.get("cursor")
        if not new_cursor or new_cursor == cursor:
            break
        cursor = new_cursor
        time.sleep(sleep_s)
    return {"pages": pages, "reviews": total, "prefix": prefix}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--app-id", type=int, required=True)
    parser.add_argument("--max-pages", type=int, default=5)
    parser.add_argument("--bucket", default="steam-reviews-raw")
    args = parser.parse_args()
    print(extract(args.app_id, args.max_pages, args.bucket))