"""Geração da base sintética (conforme enunciado) e novo caso de teste."""
import numpy as np
import pandas as pd

FEATURES = ["idade_media", "carga_sistema", "tempo_espera", "fc_media", "spo2_media",
            "pa_sistolica", "temperatura", "densidade_eventos", "eventos_24h",
            "disponibilidade_recursos"]


def gerar_base(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    df = pd.DataFrame({
        "idade_media": np.random.normal(55, 12, n).clip(18, 90),
        "carga_sistema": np.random.uniform(30, 100, n),
        "tempo_espera": np.random.normal(45, 20, n).clip(5, 180),
        "fc_media": np.random.normal(85, 15, n).clip(50, 150),
        "spo2_media": np.random.normal(95, 3, n).clip(70, 100),
        "pa_sistolica": np.random.normal(125, 20, n).clip(80, 200),
        "temperatura": np.random.normal(24, 5, n).clip(10, 40),
        "densidade_eventos": np.random.poisson(3, n),
        "eventos_24h": np.random.poisson(2, n),
        "disponibilidade_recursos": np.random.uniform(20, 100, n),
    })
    risk_score = (
        (df["carga_sistema"] > 80).astype(int) + (df["tempo_espera"] > 60).astype(int) +
        (df["spo2_media"] < 92).astype(int) + (df["fc_media"] > 110).astype(int) +
        (df["eventos_24h"] > 4).astype(int) + (df["disponibilidade_recursos"] < 40).astype(int)
    )
    df["pico_risco"] = (risk_score >= 2).astype(int)
    return df


# Atenção: no enunciado a chave aparece como "disponibilidade recursos" (com espaço);
# aqui usamos o nome correto da coluna de treino.
NOVO_CASO = {
    "idade_media": 62, "carga_sistema": 92, "tempo_espera": 75, "fc_media": 118,
    "spo2_media": 89, "pa_sistolica": 100, "temperatura": 37.5,
    "densidade_eventos": 7, "eventos_24h": 5, "disponibilidade_recursos": 25,
}
