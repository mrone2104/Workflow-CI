FROM python:3.10-slim

WORKDIR /app

# Copy file training & data
COPY modelling.py data_preprocessed.csv ./

# Install dependencies
RUN pip install --no-cache-dir \
    mlflow \
    pandas \
    scikit-learn \
    xgboost

# Jalankan retraining saat container dijalankan
CMD ["python", "modelling.py"]
