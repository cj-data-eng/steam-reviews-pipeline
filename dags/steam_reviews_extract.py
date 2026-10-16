from datetime import datetime

from airflow.decorators import dag, task


@dag(
    dag_id="steam_reviews_extract",
    start_date=datetime(2026, 10, 1),
    schedule=None,
    catchup=False,
    params={"app_id": 730, "max_pages": 3},
    tags=["steam", "raw"],
)
def steam_reviews_extract():
    @task
    def extract_reviews(**context):
        from steam_reviews.extract import extract

        p = context["params"]
        return extract(int(p["app_id"]), int(p["max_pages"]))

    extract_reviews()


steam_reviews_extract()