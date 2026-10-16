import argparse

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F


def build_spark():
    return (
        SparkSession.builder.appName("steam_reviews_silver")
        .config("spark.hadoop.fs.s3a.endpoint", "http://floci:4566")
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
        .config("spark.hadoop.fs.s3a.access.key", "test")
        .config("spark.hadoop.fs.s3a.secret.key", "test")
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
    raw = spark.read.json(f"s3a://{bucket}/raw/app_id={app_id}/*/*/*.json")

    exploded = raw.select(
        F.input_file_name().alias("source_file"),
        F.explode("reviews").alias("r"),
    )

    flat = exploded.select(
        F.col("r.recommendationid").alias("recommendation_id"),
        F.lit(app_id).alias("app_id"),
        # pseudonymised: raw steamid, name and profile URL are dropped
        F.sha2(F.col("r.author.steamid"), 256).alias("author_hash"),
        F.col("r.author.num_games_owned").alias("author_games_owned"),
        F.col("r.author.num_reviews").alias("author_num_reviews"),
        F.col("r.author.playtime_at_review").alias("playtime_at_review_min"),
        F.col("r.language").alias("language"),
        F.trim(F.col("r.review")).alias("review_text"),
        F.timestamp_seconds(F.col("r.timestamp_created")).alias("created_at"),
        F.timestamp_seconds(F.col("r.timestamp_updated")).alias("updated_at"),
        F.col("r.voted_up").alias("voted_up"),
        F.col("r.votes_up").alias("votes_up"),
        F.col("r.votes_funny").alias("votes_funny"),
        F.col("r.weighted_vote_score").cast("double").alias("weighted_vote_score"),
        F.col("r.comment_count").alias("comment_count"),
        F.col("r.steam_purchase").alias("steam_purchase"),
        F.col("r.received_for_free").alias("received_for_free"),
        F.col("source_file"),
    )

    # keep the latest version of each review (reviews can be edited)
    w = Window.partitionBy("recommendation_id").orderBy(
        F.col("updated_at").desc(), F.col("source_file").desc()
    )
    deduped = (
        flat.withColumn("rn", F.row_number().over(w)).filter("rn = 1").drop("rn")
    )

    # data quality rules: failures go to quarantine with a reason
    reason = (
        F.when(F.col("recommendation_id").isNull(), "missing_recommendation_id")
        .when(F.col("review_text").isNull() | (F.length("review_text") == 0), "empty_review_text")
        .when(F.col("created_at").isNull(), "missing_created_at")
    )
    checked = deduped.withColumn("reject_reason", reason)

    good = checked.filter("reject_reason IS NULL").drop("reject_reason")
    bad = checked.filter("reject_reason IS NOT NULL")

    good = good.withColumn("review_date", F.to_date("created_at"))

    good.write.mode("overwrite").partitionBy("review_date").parquet(
        f"s3a://{bucket}/silver/reviews/app_id={app_id}"
    )
    bad.write.mode("overwrite").parquet(
        f"s3a://{bucket}/quarantine/reviews/app_id={app_id}"
    )

    print(
        f"raw={flat.count()} deduped={deduped.count()} "
        f"silver={good.count()} quarantined={bad.count()}"
    )
    spark.stop()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--app-id", type=int, required=True)
    p.add_argument("--bucket", default="steam-reviews-raw")
    a = p.parse_args()
    main(a.app_id, a.bucket)