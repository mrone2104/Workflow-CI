import os
import pandas as pd
import mlflow
import mlflow.xgboost
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score


# ======================================================
# SET EXPERIMENT (INI KUNCI FIX ERROR 404)
# ======================================================
EXPERIMENT_NAME = "Telco_Offer"
mlflow.set_experiment(EXPERIMENT_NAME)


# ======================================================
# INFORMASI TRACKING (UNTUK DEBUG & BUKTI)
# ======================================================
print("MLFLOW_TRACKING_URI =", mlflow.get_tracking_uri())
print("MLFLOW_EXPERIMENT =", mlflow.get_experiment_by_name(EXPERIMENT_NAME))


# ======================================================
# LOAD DATA
# ======================================================
df = pd.read_csv("data_preprocessed.csv")

DROP_COLS = ["customer_id", "customer_id_encoded", "target_offer"]
X = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
y = df["target_offer_encoded"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

NUM_CLASS = y.nunique()


# ======================================================
# TRAINING & LOGGING
# ======================================================
with mlflow.start_run(run_name="XGBoost_Telco_Offer"):

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=7,
        learning_rate=0.05,
        objective="multi:softprob",
        num_class=NUM_CLASS,
        eval_metric="mlogloss",
        random_state=42
    )

    model.fit(X_train, y_train)

    # ======================
    # EVALUATION
    # ======================
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")

    # ======================
    # LOGGING METRICS
    # ======================
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("macro_f1", f1)

    # ======================
    # LOGGING PARAMETERS
    # ======================
    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("max_depth", 7)
    mlflow.log_param("learning_rate", 0.05)

    # ======================
    # LOGGING MODEL
    # ======================
    mlflow.xgboost.log_model(
        model,
        artifact_path="model"
    )

    # ======================
    # SIMPAN RUN ID (UNTUK DOCKER BUILD)
    # ======================
    run_id = mlflow.active_run().info.run_id
    with open("run_id.txt", "w") as f:
        f.write(run_id)

    print("RUN_ID =", run_id)
    print(f"Accuracy = {acc:.4f}")
    print(f"Macro F1 = {f1:.4f}")


print("TRAINING SELESAI")
