"""Prédiction avec le modèle entraîné (models/model.joblib)."""
from pathlib import Path

import joblib
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "model.joblib"
COLONNES = ["duration", "dst_port", "src_bytes", "dst_bytes", "nb_packets",
            "nb_failed_logins", "nb_conn_same_host", "syn_rate", "protocol"]

_model = None


def predict(df: pd.DataFrame) -> list:
    """df : mêmes colonnes que flux_train.csv, sans 'attaque'.
    Retourne une liste de 0 (normal) ou 1 (attaque), une valeur par ligne."""
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    X = df.copy()
    if "protocol" in X.columns:
        X["protocol"] = X["protocol"].astype(str).str.upper()
    return [int(p) for p in _model.predict(X[COLONNES])]


if __name__ == "__main__":
    data = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "flux_train.csv")
    preds = predict(data.drop(columns=["attaque"]).head(10))
    print(preds)
