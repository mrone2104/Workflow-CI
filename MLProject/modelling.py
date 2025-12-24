import pandas as pd
import mlflow
import mlflow.xgboost
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score


# ======================================================
# LOAD DATA
# ======================================================
DATA_PATH = "data_preprocessed.csv"
df = pd.read_csv(DATA_PATH)

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
# START MLFLOW RUN (SAFE FOR PROJECT & LOCAL)
# ======================================================
# Jika dijalankan via `mlflow run MLProject`,
# parent run sudah ada → kita pakai nested run
# Jika dijalankan manual (VS Code), buat run baru
if mlflow.active_run() is None:
    run_ctx = mlflow.start_run(nested=True)
else:
    run_ctx = None


try:
    # ==================================================
    # TRAIN MODEL
    # ==================================================
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

    # ==================================================
    # EVALUATION
    # ==================================================
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")

    # ==================================================
    # LOG TO MLFLOW
    # ==================================================
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("macro_f1", f1)

    # ⬅️ WAJIB agar bisa build Docker
    mlflow.xgboost.log_model(
        model,
        artifact_path="model"
    )

    print("=== TRAINING SELESAI ===")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Macro F1  : {f1:.4f}")

finally:
    # Tutup run hanya jika kita yang membuka
    if run_ctx is not None:
        mlflow.end_run()
