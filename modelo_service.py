"""Serviço de predição: carrega o modelo salvo e expõe predizer()."""
from functools import lru_cache
from pathlib import Path
import joblib
import pandas as pd
from dados import FEATURES


@lru_cache(maxsize=1)
def _carregar():
    return joblib.load(Path(__file__).resolve().parent / "models" / "modelo_risco.joblib")


def predizer(dados: dict) -> float:
    X = pd.DataFrame([dados])[FEATURES]
    return float(_carregar().predict_proba(X)[0, 1])
