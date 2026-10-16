import argparse
import io
import json
import os
import re
from importlib.metadata import version

import boto3
import joblib
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, DoubleType, StringType, StructField, StructType

MODEL_NAME = "review_recommend_tfidf_lr"

THEMES = {
    "cheating": [r"\bcheat", r"\bhack", r"aimbot", r"wallhack", r"\bvac\b", r"anti-?cheat", r"\bsmurf"],
    "matchmaking": [r"matchmaking", r"\bqueue", r"\brank", r"premier", r"\belo\b", r"faceit", r"teammate"],
    "performance": [r"\bfps\b", r"\blag", r"\bcrash", r"stutter", r"optimi", r"performance", r"\bbug", r"\bserver", r"\btick"],
    "skins_and_price": [r"\bskin", r"\bcase", r"\bprice", r"market", r"\btrade", r"free to play", r"\bf2p\b", r"\bpay"],
    "gameplay": [r"gameplay", r"gunplay", r"\bmaps?\b", r"recoil", r"\bspray", r"\bweapon", r"\baim\b"],
    "community": [r"toxic", r"communit", r"\breport", r"\bban", r"voice chat", r"\bchat\b"],
}
PATTERNS = {t: re.compile("|".join(p), re.IGNORECASE) for t, p in THEMES.items()}

_ANALYZER = None


def enrich(text):
    global _ANALYZER
    if _ANALYZER is None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        _ANALYZER = SentimentIntensityAnalyzer()
    text = text or ""
    score = float(_ANALYZER.polarity_scores(text)["compound"])
    themes = [t for t, rx in PATTERNS.items() if rx.search(text)]
    return (score, themes)


ENRICH_SCHEMA = StructType(
    [
        StructField("compound", DoubleType()),
        StructField("themes", ArrayType(StringType())),
    ]
)


def load_latest_model(bucket):
    s3 = boto3.client("s3", endpoint_url=os.environ.get("S3_ENDPOINT_URL"))
    latest = json.loads(
        s3.get_object(Bucket=bucket, Key=f"models/{MODEL_NAME}/latest.json")["Body"].read()
    )
    ver = latest["version"]
    body = s3.get_object(Bucket=bucket, Key=f"models/{MODEL_NAME}/{ver}/model.joblib")["Body"].read()
    return joblib.load(io.BytesIO(body)), ver


def build_spark():
    return (
        SparkSession.builder.appName("steam_reviews_gold")
        .config("spark.hadoop.fs.s3a.endpoint", "http://floci:4566")
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
        .config("spark.hadoop.fs.s3a.access.key", "test")
        .config("spark.hadoop.fs.s3a.secret.key", "test")
        # disabled for the local emulator only; keep the default on real AWS
        .config("spark.hadoop.fs.s3a.change.detection.mode", "none")
        .config(
            "spark.hadoop.fs.s3a.aws.credentials.provider",
            "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider",
        )
        .config("spark.sql.sources.partitionOverwriteMode", "dynamic")
        .config("spark.sql.parquet.outputTimestampType", "TIMESTAMP_MICROS")
        .getOrCreate()
    )


def main(app_id, bucket):
    spark = build_spark()
    enrich_udf = F.udf(enrich, ENRICH_SCHEMA)

    model, model_ver = load_latest_model(bucket)
    bc_model = spark.sparkContext.broadcast(model)

    @F.pandas_udf("double")
    def recommend_prob(texts: pd.Series) -> pd.Series:
        return pd.Series(bc_model.value.predict_proba(texts.fillna("").astype(str))[:, 1])

    silver = spark.read.parquet(f"s3a://{bucket}/silver/reviews/app_id={app_id}")

    enriched = (
        silver.withColumn("e", enrich_udf("review_text"))
        .withColumn("sentiment_score", F.col("e.compound"))
        .withColumn("themes", F.col("e.themes"))
        .drop("e")
        .withColumn(
            "sentiment_label",
            F.when(F.col("sentiment_score") >= 0.05, "positive")
            .when(F.col("sentiment_score") <= -0.05, "negative")
            .otherwise("neutral"),
        )
        .withColumn("model_name", F.lit("vader"))
        .withColumn("model_version", F.lit(version("vaderSentiment")))
        .withColumn("recommend_prob", recommend_prob("review_text"))
        .withColumn("recommend_model", F.lit(MODEL_NAME))
        .withColumn("recommend_model_version", F.lit(model_ver))
        .withColumn("enriched_at", F.current_timestamp())
    )
    enriched.cache()

    enriched.write.mode("overwrite").partitionBy("review_date").parquet(
        f"s3a://{bucket}/gold/reviews_enriched/app_id={app_id}"
    )

    by_theme = enriched.withColumn(
        "theme",
        F.explode(
            F.when(F.size("themes") == 0, F.array(F.lit("other"))).otherwise(F.col("themes"))
        ),
    )
    daily = by_theme.groupBy("review_date", "theme").agg(
        F.count("*").alias("review_count"),
        F.round(F.avg("sentiment_score"), 3).alias("avg_sentiment"),
        F.round(F.avg("recommend_prob"), 3).alias("avg_recommend_prob"),
        F.round(F.avg(F.col("voted_up").cast("int")), 3).alias("pct_recommended"),
    )
    daily.write.mode("overwrite").parquet(
        f"s3a://{bucket}/gold/daily_theme_sentiment/app_id={app_id}"
    )

    total = enriched.count()
    themed = enriched.filter(F.size("themes") > 0).count()
    print(f"gold rows={total} themed={themed} recommend_model_version={model_ver}")
    spark.stop()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--app-id", type=int, required=True)
    p.add_argument("--bucket", default="steam-reviews-raw")
    a = p.parse_args()
    main(a.app_id, a.bucket)