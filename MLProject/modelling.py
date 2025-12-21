import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import xgboost as xgb

# =============================
# 1. LOAD DATASET (PREPROCESSED)
# =============================
df = pd.read_csv("data_preprocessed.csv")

# =============================
# 2. FEATURE & TARGET
# =============================
drop_cols = [
    "customer_id",
    "customer_id_encoded",
    "target_offer"
]

X = df.drop(columns=[c for c in drop_cols if c in df.columns])
y = df["target_offer_encoded"].astype(int)

# =============================
# 3. SPLIT DATA
# =============================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# =============================
# 4. MLFLOW AUTOLOG
# =============================
# PENTING:
# - JANGAN pakai mlflow.start_run()
# - JANGAN pakai mlflow.set_experiment()
# karena lifecycle run diatur oleh `mlflow run`
mlflow.sklearn.autolog()

# =============================
# 5. MODEL TRAINING
# =============================
model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=7,
    learning_rate=0.05,
    objective="multi:softprob",
    num_class=y.nunique(),
    eval_metric="mlogloss",
    random_state=42
)

model.fit(X_train, y_train)

# =============================
# 6. EVALUATION
# =============================
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="macro")

# =============================
# 7. MANUAL LOGGING
# =============================
mlflow.log_metric("accuracy_manual", acc)
mlflow.log_metric("macro_f1_manual", f1)

# =============================
# 8. OUTPUT
# =============================
print("Accuracy:", acc)
print("Macro F1:", f1)

# =============================
# 9. SIMPAN RUN_ID (WAJIB UNTUK DOCKER)
# =============================
run_id = mlflow.active_run().info.run_id

with open("run_id.txt", "w") as f:
    f.write(run_id)

print(f"RUN_ID={run_id}")
