from datetime import datetime

from airflow.decorators import dag, task
from airflow.operators.bash import BashOperator


@dag(
    dag_id="steam_reviews_extract",
    start_date=datetime(2026, 10, 1),
    schedule=None,
    catchup=False,
    params={"app_id": 730, "max_pages": 3},
    tags=["steam", "raw", "silver"],
)
def steam_reviews_extract():
    @task
    def extract_reviews(**context):
        from steam_reviews.extract import extract

        p = context["params"]
        return extract(int(p["app_id"]), int(p["max_pages"]))

    spark_silver = BashOperator(
        task_id="spark_silver",
        bash_command=(
            "spark-submit --master 'local[*]' "
            "--packages org.apache.hadoop:hadoop-aws:3.3.4 "
            "--conf spark.jars.ivy=/tmp/ivy "
            "/opt/airflow/spark_jobs/silver_reviews.py "
            "--app-id {{ params.app_id }}"
        ),
    )

    spark_gold = BashOperator(
        task_id="spark_gold",
        bash_command=(
            "spark-submit --master 'local[*]' "
            "--packages org.apache.hadoop:hadoop-aws:3.3.4 "
            "--conf spark.jars.ivy=/tmp/ivy "
            "/opt/airflow/spark_jobs/gold_reviews.py "
            "--app-id {{ params.app_id }}"
        ),
    )

    extract_reviews() >> spark_silver >> spark_gold


steam_reviews_extract()