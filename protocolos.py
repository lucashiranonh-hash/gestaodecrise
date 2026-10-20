"""Base de protocolos + avaliação de quais são acionados pelos dados do paciente."""

PROTOCOLOS = [
    "Se SPO2 < 88% → iniciar suporte respiratório imediato",
    "Se carga do sistema > 85% → abrir unidades temporárias",
    "Se FC > 120 → iniciar monitoramento intensivo",
    "Se eventos_24h > 5 → ativar equipe de resposta rápida",
    "Se disponibilidade_recursos < 30% → redistribuir insumos",
    "Se tempo_espera > 90 min → priorizar triagem emergencial",
    "Se temperatura > 38°C média → iniciar protocolo infeccioso",
    "Se densidade_eventos > 6 → ativar plano de contingência",
    "Se PA sistólica < 90 → iniciar protocolo de choque",
    "Se risco sistêmico alto → ativar estado de emergência",
]

# Cada regra: (índice do protocolo, condição sobre os dados)
_REGRAS = [
    (0, lambda d, p: d["spo2_media"] < 88),
    (1, lambda d, p: d["carga_sistema"] > 85),
    (2, lambda d, p: d["fc_media"] > 120),
    (3, lambda d, p: d["eventos_24h"] > 5),
    (4, lambda d, p: d["disponibilidade_recursos"] < 30),
    (5, lambda d, p: d["tempo_espera"] > 90),
    (6, lambda d, p: d["temperatura"] > 38),
    (7, lambda d, p: d["densidade_eventos"] > 6),
    (8, lambda d, p: d["pa_sistolica"] < 90),
    (9, lambda d, p: p >= 0.7),  # risco sistêmico alto
]


def classificar_risco(prob: float) -> str:
    return "ALTO" if prob >= 0.7 else "MODERADO" if prob >= 0.4 else "BAIXO"


def protocolos_acionados(dados: dict, prob: float) -> list[str]:
    return [PROTOCOLOS[i] for i, regra in _REGRAS if regra(dados, prob)]
