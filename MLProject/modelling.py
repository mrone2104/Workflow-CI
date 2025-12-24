import pandas as pd
import mlflow
import mlflow.xgboost
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from mlflow.tracking import MlflowClient

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
# TRAIN MODEL (CI SAFE)
# ======================================================
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

# ======================================================
# EVALUATION
# ======================================================
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="macro")

# ======================================================
# LOG TO MLFLOW (AUTO RUN)
# ======================================================
mlflow.log_metric("accuracy", acc)
mlflow.log_metric("macro_f1", f1)

mlflow.xgboost.log_model(
    model,
    artifact_path="model"
)

print("=== TRAINING SELESAI ===")
print(f"Accuracy : {acc:.4f}")
print(f"Macro F1 : {f1:.4f}")

# ======================================================
# SIMPAN RUN_ID TERBARU (UNTUK CI)
# ======================================================
client = MlflowClient()
experiment = client.get_experiment_by_name("Default")

runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["attributes.start_time DESC"],
    max_results=1
)

latest_run_id = runs[0].info.run_id

with open("latest_run.txt", "w") as f:
    f.write(latest_run_id)

print("Latest RUN_ID saved:", latest_run_id)
