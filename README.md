# Projet 2 — Détecter des attaques réseau avec le Machine Learning

**Auteur** : El Hadj Ibrihima Dienne

## Lancer le projet
```bash
pip install -r requirements.txt
python src/train.py      # compare les modèles, enregistre models/model.joblib
python src/predict.py    # petit exemple de prédiction
```
Utilisation de `predict` :
```python
import pandas as pd
from src.predict import predict
df = pd.read_csv("data/flux_train.csv").drop(columns=["attaque"])
print(predict(df)[:10])
```
> Le modèle a été enregistré avec les versions de `requirements.txt` (scikit-learn 1.8.0). Utilisez les mêmes versions pour le recharger.

## Exploration des données
3000 flux, 10 colonnes. Seule `duration` a des valeurs manquantes (90, soit 3 %). Classes déséquilibrées : 72,2 % de flux normaux et 27,8 % d'attaques. Les attaques ont en moyenne un `dst_bytes` très faible (533 contre 4 706), plus de connexions vers le même hôte (30 contre 5), un `syn_rate` plus élevé (0,44 contre 0,12) et plus d'échecs d'authentification (3,3 contre 0,16).

## Méthode
Un `Pipeline` scikit-learn contient le prétraitement (imputation par la médiane + `StandardScaler` pour les variables numériques, `OneHotEncoder` pour `protocol`) et le modèle. Séparation 80 % / 20 % stratifiée (600 flux de validation), `random_state=42`. Les modèles à classes déséquilibrées utilisent `class_weight="balanced"`. Le F1 en validation croisée (5 plis, sur toutes les données) confirme le classement.

## Résultats

**Nom / Prénom :** Dienne El Hadj Ibrihima

| Modèle | F1 (attaque) | Rappel (attaque) | Précision (attaque) | F1 validation croisée (5 plis) |
|---|---|---|---|---|
| Régression logistique | 0,920 | 0,898 | 0,943 | 0,896 |
| Arbre de décision (profondeur 6) | 0,953 | 0,964 | 0,942 | 0,935 |
| **Forêt aléatoire (300 arbres)** | **0,982** | **0,976** | **0,988** | **0,960** |
| Gradient Boosting | 0,973 | 0,970 | 0,976 | 0,953 |

**Modèle retenu et pourquoi :** la forêt aléatoire. Elle a le meilleur F1 sur la validation (0,982) et en validation croisée (0,960), avec à la fois le meilleur rappel et la meilleure précision. Les relations entre variables ne sont pas linéaires (les attaques combinent plusieurs signaux), ce qui explique l'écart avec la régression logistique. Le modèle final est ré-entraîné sur les 3000 flux et enregistré dans `models/model.joblib`. Variables les plus utiles : `dst_bytes` (0,36), `nb_conn_same_host` (0,23), `syn_rate` (0,13) et `nb_failed_logins` (0,10).

**Matrice de confusion (forêt aléatoire, validation, 600 flux) :**

| | Prédit normal | Prédit attaque |
|---|---|---|
| **Réel normal** | 431 (VN) | 2 (FP) |
| **Réel attaque** | 4 (FN) | 163 (VP) |

**Analyse des erreurs (faux négatifs / faux positifs) :**
- **4 faux négatifs** : 4 attaques sur 167 n'ont pas été détectées (2,4 %). Ces flux ressemblent à du trafic normal : aucun échec d'authentification, peu de connexions vers le même hôte (3 à 7), `syn_rate` faible, ports courants (80, 22, 443, 8080). Ce sont des attaques discrètes (« low and slow »).
- Un faux négatif est le pire cas en sécurité : une attaque non détectée passe sans alerte et peut mener à une intrusion, une fuite de données ou un déni de service sans que personne ne réagisse.
- **2 faux positifs** : 2 flux normaux ont été signalés à tort. Un flux très court avec beaucoup de connexions vers le même hôte et un `syn_rate` élevé, et une connexion SSH avec 9 échecs d'authentification. Cela coûte une fausse alerte à vérifier, ce qui est moins grave qu'une attaque manquée.
- **Pistes d'amélioration** : abaisser le seuil de décision pour augmenter le rappel (au prix de plus de fausses alertes), ajouter des variables (par ex. rapport `src_bytes / dst_bytes`), ou régler les hyperparamètres.
