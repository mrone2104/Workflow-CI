import pandas as pd
import mlflow
import mlflow.xgboost
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from mlflow.models import infer_signature

# =========================
# 1. LOAD DATA
# =========================
df = pd.read_csv("data_preprocessed.csv")

DROP_COLS = ["customer_id", "customer_id_encoded", "target_offer"]
X = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
y = df["target_offer_encoded"].astype(int)

# =========================
# 2. TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

NUM_CLASS = y.nunique()

# =========================
# 3. TRAIN + LOG WITH MLFLOW
# =========================
with mlflow.start_run():

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

    # =========================
    # 4. EVALUATION
    # =========================
    y_pred = model.predict(X_test)

    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
    mlflow.log_metric(
        "f1_score",
        f1_score(y_test, y_pred, average="weighted")
    )

    # =========================
    # 5. SIGNATURE & INPUT EXAMPLE (WAJIB UNTUK SERVING)
    # =========================
    input_example = X_train.head(1)
    signature = infer_signature(
        X_train,
        model.predict_proba(X_train)
    )

    # =========================
    # 6. LOG MODEL (INFERENCE READY)
    # =========================
    mlflow.xgboost.log_model(
        model,
        artifact_path="model",
        signature=signature,
        input_example=input_example
    )

print("✅ Training finished & model logged correctly")
