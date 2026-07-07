import mlflow
import mlflow.sklearn
import joblib
import os
from tabpfn_wrapper import BalancedBaggingTabPFNBinary

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, '..', 'model_1-X2.pkl')
model_b_path = os.path.join(BASE_DIR, '..', 'model_X-2.pkl')
model = joblib.load(model_path)
model_B = joblib.load(model_b_path)

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("LaLiga_Prediction")

with mlflow.start_run():
    mlflow.sklearn.log_model(
        model,
        artifact_path = "model",
        registered_model_name = "LaLiga_RF_1X2"
    )

with mlflow.start_run():
    mlflow.sklearn.log_model(
        model_B,
        artifact_path = "model",
        registered_model_name = "LaLiga_TabPFN_X2"
    )