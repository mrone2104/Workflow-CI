import pandas as pd
import mlflow
import mlflow.xgboost
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# ==========================================
# 0. MLFLOW CONFIG
# ==========================================
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Telco_Offer_Basic")

# ==========================================
# 1. LOAD DATASET
# ==========================================
df = pd.read_csv("data_preprocessed.csv")

# ==========================================
# 2. FEATURE & TARGET
# ==========================================
DROP_COLS = ["customer_id", "customer_id_encoded", "target_offer"]
X = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
y = df["target_offer_encoded"].astype(int)

# ==========================================
# 3. TRAIN TEST SPLIT
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

NUM_CLASS = y.nunique()

# ==========================================
# 4. TRAINING + LOGGING
# ==========================================
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

# ==========================================
# 5. EVALUATION
# ==========================================
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="macro")

mlflow.log_metric("accuracy", acc)
mlflow.log_metric("macro_f1", f1)

# ==========================================
# 6. LOG MODEL
# ==========================================
mlflow.xgboost.log_model(model, artifact_path="model")

# ==========================================
# 7. SIMPAN RUN_ID
# ==========================================
run = mlflow.active_run()
if run:
    run_id = run.info.run_id
    with open("run_id.txt", "w") as f:
        f.write(run_id)

print("=== TRAINING SELESAI ===")
print("Accuracy:", acc)
print("Macro F1:", f1)

# ==========================================
# END OF FILE
# ==========================================
