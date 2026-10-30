"""Entraînement d'un détecteur d'attaques réseau (IDS).

Usage : python src/train.py
- explore les données
- compare plusieurs modèles sur un jeu de validation (F1 de la classe attaque)
- ré-entraîne le meilleur sur toutes les données
- enregistre le Pipeline complet dans models/model.joblib
"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data" / "flux_train.csv"
MODEL_PATH = BASE / "models" / "model.joblib"
SEED = 42

NUM = ["duration", "dst_port", "src_bytes", "dst_bytes", "nb_packets",
       "nb_failed_logins", "nb_conn_same_host", "syn_rate"]
CAT = ["protocol"]


def construire_pipeline(modele):
    """Prétraitement (imputation, échelle, encodage) + modèle dans un Pipeline."""
    prepro = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                          ("scaler", StandardScaler())]), NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
    ])
    return Pipeline([("prepro", prepro), ("model", modele)])


def explorer(df):
    print("Taille :", df.shape)
    print("\nValeurs manquantes :\n", df.isna().sum()[df.isna().sum() > 0])
    print("\nRépartition de 'attaque' :\n", df["attaque"].value_counts(normalize=True).round(3))


def main():
    df = pd.read_csv(DATA)
    explorer(df)
    X, y = df[NUM + CAT], df["attaque"]
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)

    modeles = {
        "Régression logistique": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Arbre de décision": DecisionTreeClassifier(max_depth=6, class_weight="balanced",
                                                    random_state=SEED),
        "Forêt aléatoire": RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                                  random_state=SEED, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=SEED),
    }

    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    scores = {}
    for nom, m in modeles.items():
        pipe = construire_pipeline(m)
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_val)
        cv_f1 = cross_val_score(construire_pipeline(m), X, y, cv=cv, scoring="f1").mean()
        scores[nom] = f1_score(y_val, pred)
        tn, fp, fn, tp = confusion_matrix(y_val, pred).ravel()
        print(f"\n=== {nom} ===")
        print(classification_report(y_val, pred, target_names=["normal", "attaque"], digits=3))
        print(f"Matrice de confusion [[TN FP][FN TP]] : [[{tn} {fp}] [{fn} {tp}]]")
        print(f"F1={f1_score(y_val, pred):.3f}  Rappel={recall_score(y_val, pred):.3f}  "
              f"Précision={precision_score(y_val, pred):.3f}  F1 (CV 5 plis)={cv_f1:.3f}")

    meilleur = max(scores, key=scores.get)
    print(f"\nModèle retenu : {meilleur} (F1 validation = {scores[meilleur]:.3f})")

    final = construire_pipeline(modeles[meilleur])
    final.fit(X, y)  # ré-entraînement sur toutes les données
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(final, MODEL_PATH)
    print("Modèle enregistré :", MODEL_PATH)


if __name__ == "__main__":
    main()
