import argparse
import re
from importlib.metadata import version

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, DoubleType, StringType, StructField, StructType

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

    silver = spark.read.parquet(f"s3a://{bucket}/silver/reviews/app_id={app_id}")
    score = F.col("e.compound")

    enriched = (
        silver.withColumn("e", enrich_udf("review_text"))
        .withColumn("sentiment_score", score)
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
        F.round(F.avg(F.col("voted_up").cast("int")), 3).alias("pct_recommended"),
    )
    daily.write.mode("overwrite").parquet(
        f"s3a://{bucket}/gold/daily_theme_sentiment/app_id={app_id}"
    )

    total = enriched.count()
    themed = enriched.filter(F.size("themes") > 0).count()
    scored = enriched.filter("sentiment_label != 'neutral'")
    n_scored = scored.count()
    agree = scored.filter((F.col("sentiment_label") == "positive") == F.col("voted_up")).count()
    rate = round(agree / n_scored, 3) if n_scored else None
    print(f"gold rows={total} themed={themed} non_neutral={n_scored} agreement_with_voted_up={rate}")
    spark.stop()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--app-id", type=int, required=True)
    p.add_argument("--bucket", default="steam-reviews-raw")
    a = p.parse_args()
    main(a.app_id, a.bucket)