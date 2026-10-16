import io
import json
import os
from datetime import datetime, timezone

import boto3
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline

MODEL_NAME = "review_recommend_tfidf_lr"
MIN_ROWS = 500
MIN_TEST_AUC = 0.75


def get_s3():
    return boto3.client("s3", endpoint_url=os.environ.get("S3_ENDPOINT_URL"))


def read_silver(s3, bucket, app_id):
    prefix = f"silver/reviews/app_id={app_id}/"
    cols = ["recommendation_id", "review_text", "voted_up", "created_at"]
    frames = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".parquet"):
                body = s3.get_object(Bucket=bucket, Key=obj["Key"])["Body"].read()
                frames.append(pd.read_parquet(io.BytesIO(body), columns=cols))
    if not frames:
        raise RuntimeError(f"no silver files under s3://{bucket}/{prefix}")
    return pd.concat(frames, ignore_index=True)


def train(app_id, bucket="steam-reviews-raw"):
    s3 = get_s3()
    df = read_silver(s3, bucket, app_id).sort_values("created_at").reset_index(drop=True)
    if len(df) < MIN_ROWS:
        raise ValueError(f"only {len(df)} rows in silver, need at least {MIN_ROWS}")

    y = df["voted_up"].astype(int)
    cut = int(len(df) * 0.8)
    train_df, test_df = df.iloc[:cut], df.iloc[cut:]
    y_train, y_test = y.iloc[:cut], y.iloc[cut:]

    pipe = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    pipe.fit(train_df["review_text"], y_train)

    proba = pipe.predict_proba(test_df["review_text"])[:, 1]
    pred = (proba >= 0.5).astype(int)
    auc = float(roc_auc_score(y_test, proba))

    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    metrics = {
        "model": MODEL_NAME,
        "version": version,
        "app_id": app_id,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_window": [str(train_df.created_at.min()), str(train_df.created_at.max())],
        "test_window": [str(test_df.created_at.min()), str(test_df.created_at.max())],
        "test_auc": round(auc, 4),
        "test_accuracy": round(float((pred == y_test).mean()), 4),
        "always_positive_accuracy": round(float(y_test.mean()), 4),
        "negative_class_precision": round(float(precision_score(y_test, pred, pos_label=0)), 4),
        "negative_class_recall": round(float(recall_score(y_test, pred, pos_label=0)), 4),
        "min_auc_gate": MIN_TEST_AUC,
    }

    prefix = f"models/{MODEL_NAME}/{version}"
    buf = io.BytesIO()
    joblib.dump(pipe, buf)
    s3.put_object(Bucket=bucket, Key=f"{prefix}/model.joblib", Body=buf.getvalue())
    s3.put_object(Bucket=bucket, Key=f"{prefix}/metrics.json", Body=json.dumps(metrics, indent=2).encode())

    if auc < MIN_TEST_AUC:
        raise ValueError(f"test AUC {auc:.3f} is below the gate {MIN_TEST_AUC}; model {version} not promoted")

    s3.put_object(
        Bucket=bucket,
        Key=f"models/{MODEL_NAME}/latest.json",
        Body=json.dumps({"version": version}).encode(),
    )
    return metrics