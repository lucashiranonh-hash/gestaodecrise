"""Parte 1: treino, avaliação e salvamento do modelo preditivo de pico de risco."""
from pathlib import Path
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             ConfusionMatrixDisplay, confusion_matrix)
from sklearn.model_selection import train_test_split, cross_val_score

from dados import FEATURES, NOVO_CASO, gerar_base

BASE = Path(__file__).resolve().parent
MODELS = BASE / "models"
MODELS.mkdir(exist_ok=True)  # cria a pasta se não existir

df = gerar_base()
print("Distribuição da classe alvo:\n", df["pico_risco"].value_counts(normalize=True).round(3))

X, y = df[FEATURES], df["pico_risco"]
# stratify mantém a proporção das classes em treino e teste
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

modelo = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
modelo.fit(X_tr, y_tr)

pred = modelo.predict(X_te)
print(f"\nAcurácia (teste): {accuracy_score(y_te, pred):.4f}")
print(f"Acurácia CV 5-fold: {cross_val_score(modelo, X, y, cv=5).mean():.4f}")
print("\nMatriz de confusão:\n", confusion_matrix(y_te, pred))
print("\n", classification_report(y_te, pred, target_names=["sem pico", "pico"]))

ConfusionMatrixDisplay.from_predictions(y_te, pred, display_labels=["sem pico", "pico"])
plt.title("Matriz de confusão - RandomForest")
plt.savefig(MODELS / "matriz_confusao.png", dpi=120, bbox_inches="tight")

imp = pd.Series(modelo.feature_importances_, index=FEATURES).sort_values(ascending=False)
print("\nImportância das variáveis:\n", imp.round(3))

joblib.dump(modelo, MODELS / "modelo_risco.joblib")
print(f"\nModelo salvo em {MODELS / 'modelo_risco.joblib'}")

prob = modelo.predict_proba(pd.DataFrame([NOVO_CASO])[FEATURES])[0, 1]
print(f"Simulação novo paciente -> probabilidade de pico de risco: {prob:.2f}")
