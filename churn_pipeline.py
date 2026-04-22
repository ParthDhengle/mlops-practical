from __future__ import annotations

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from prefect import flow, task


# ─────────────────────────────────────────────
# TASKS  (equivalent to your PythonOperators)
# ─────────────────────────────────────────────

@task(name="Load & Preprocess", retries=1, retry_delay_seconds=120)
def load_and_preprocess(csv_path: str) -> dict:
    df = pd.read_csv(csv_path)
    df = df.drop(columns=["RowNumber", "CustomerId", "Surname"])
    df = pd.get_dummies(df, columns=["Geography", "Gender"], drop_first=True)

    X = df.drop(columns=["Exited"])
    y = df["Exited"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    print(f"✅ Data loaded and preprocessed. Shape: {X_train.shape}")

    # Return dict instead of XCom push — clean & Pythonic
    return {
        "X_train": X_train.tolist(),
        "X_test":  X_test.tolist(),
        "y_train": y_train.tolist(),
        "y_test":  y_test.tolist(),
    }


@task(name="Train Model", retries=1, retry_delay_seconds=120)
def train_model(data: dict) -> str:
    X_train = np.array(data["X_train"])
    y_train = np.array(data["y_train"])

    n = 250
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("churn-experiment")

    with mlflow.start_run(run_name=f"RF_n{n}") as run:
        model = RandomForestClassifier(n_estimators=n, random_state=42)
        model.fit(X_train, y_train)
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("n_estimators", n)
        mlflow.sklearn.log_model(model, "model")
        run_id = run.info.run_id

    print(f"✅ Model trained. Run ID: {run_id}")
    return run_id  # passed directly to next task — no XCom needed


@task(name="Evaluate Model", retries=1, retry_delay_seconds=120)
def evaluate_model(data: dict, run_id: str) -> float:
    X_test = np.array(data["X_test"])
    y_test = np.array(data["y_test"])

    mlflow.set_tracking_uri("http://localhost:5000")

    model  = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)

    with mlflow.start_run(run_id=run_id):
        mlflow.log_metric("accuracy", acc)

    print(f"✅ Accuracy: {acc:.4f}")
    return acc


# ─────────────────────────────────────────────
# FLOW  (equivalent to your DAG)
# ─────────────────────────────────────────────

@flow(
    name="Churn ML Pipeline",
    description="Load → Preprocess → Train → Evaluate with MLflow",
)
def churn_ml_pipeline(csv_path: str = "data/Churn_Modelling.csv"):
    data   = load_and_preprocess(csv_path)   # t1
    run_id = train_model(data)               # t2  (receives t1 output directly)
    acc    = evaluate_model(data, run_id)    # t3  (receives t1 + t2 outputs)
    print(f"\n🏁 Pipeline complete! Final accuracy: {acc:.4f}")


# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # ── Option 1: Run immediately ──────────────────────────────────────────
    churn_ml_pipeline()

    # ── Option 2: Schedule (every Monday 9 AM, like your Airflow cron) ─────
    # Uncomment the block below and comment out the line above to schedule it.
    #
    # from prefect.client.schemas.schedules import CronSchedule
    # churn_ml_pipeline.serve(
    #     name="churn-weekly-run",
    #     schedules=[CronSchedule(cron="0 9 * * 1", timezone="Asia/Kolkata")],
    # )