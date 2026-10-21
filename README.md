# CardioIA – Fase 6: Sistema Preditivo Multiagente

**Integrantes:** Nome 1 (RM...), Nome 2 (RM...), ...
**Vídeo demonstrativo:** <>

## Objetivo
Prever picos de risco cardíaco (`pico_risco`) com ML sobre base sintética e integrar o modelo a agentes
(OpenAI Agents SDK) que cruzam o risco com a base de protocolos e geram recomendação estruturada.

## Estrutura
- `dados.py` – base sintética e novo caso de teste
- `train_model.py` – treino (RandomForest), avaliação e salvamento em `models/modelo_risco.joblib`
- `modelo_service.py` – carrega o modelo e prediz
- `protocolos.py` – base de protocolos e regras de acionamento
- `agentes.py` – agentes, tools, handoffs, guardrail e execução
- `notebook/` – notebook do Colab; `docs/` – relatório técnico e PDF de arquitetura

## Execução
```bash
pip install -r requirements.txt
python train_model.py
export OPENAI_API_KEY=sua_chave      # Windows: set OPENAI_API_KEY=...
python agentes.py
```

## Arquitetura
Orquestrador → (handoff) Analista de Risco [tool `prever_risco`] → (handoff) Especialista em Protocolos
[tool `consultar_protocolos`] → (handoff) Planejador de Resposta [saída `RespostaFinal` + guardrail].

## Modelo
RandomForest, split 80/20 estratificado, métricas: acurácia e matriz de confusão (`models/matriz_confusao.png`).
Limitação: o alvo é gerado por regras determinísticas, então métricas altas refletem a base sintética, não o mundo real.
