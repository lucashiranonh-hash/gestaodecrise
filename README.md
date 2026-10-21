# FIAP - Faculdade de Informática e Administração Paulista

# 🚀 FASE 6 — Sistema Preditivo Multiagente para Eventos Cardíacos (CardioIA)

📚 Graduação ON em Inteligência Artificial

## 👥 Integrantes

| Nome | RM |
|---|---|
| Lucas Yuji Nakayama Hirano | 563420 |

## 🎥 Vídeo demonstrativo

**Link <>

---

## 👩🏻‍💻 Sobre esta Fase

Nesta fase o CardioIA deixa de atuar apenas de forma reativa e passa a operar de maneira **preditiva e orientada a risco**. O projeto integra um modelo de Machine Learning a uma arquitetura de **agentes especializados** (OpenAI Agents SDK), que colaboram para prever o risco, cruzá-lo com uma base de protocolos e gerar uma recomendação final estruturada.

## 🎯 Objetivo da Fase

- Treinar um modelo supervisionado para prever picos de risco cardíaco (`pico_risco`) em uma base sintética.
- Integrar o modelo a um sistema multiagente, compartilhando o resultado entre os agentes.
- Cruzar o risco previsto com uma base simulada de protocolos.
- Gerar uma resposta final com probabilidade, classificação de risco e protocolos sugeridos.

## 📖 Conteúdos Abordados

- Geração de dados sintéticos e preparação de dados
- Classificação supervisionada, separação treino/teste e validação cruzada
- Métricas de avaliação (acurácia, matriz de confusão, precisão, recall, F1)
- Persistência de modelos com `joblib`
- Agentes, tools, handoffs e histórico de mensagens
- Validação de saída (Pydantic e guardrails)

## 🛠 Tecnologias Utilizadas

Python · Pandas / NumPy · Scikit-Learn (Random Forest) · Joblib · Matplotlib · OpenAI Agents SDK · Pydantic

---

## 📂 Projeto — CardioIA: Sistema Preditivo Multiagente

### Base de dados

Base sintética de **5.000 registros**, gerada em `dados.py` (`np.random.seed(42)`), com 10 variáveis: `idade_media`, `carga_sistema`, `tempo_espera`, `fc_media`, `spo2_media`, `pa_sistolica`, `temperatura`, `densidade_eventos`, `eventos_24h` e `disponibilidade_recursos`. O alvo `pico_risco` vale 1 quando pelo menos 2 de 6 condições críticas são atendidas (carga > 80, espera > 60, SpO2 < 92, FC > 110, eventos_24h > 4, recursos < 40). A classe positiva representa cerca de 26,6% dos casos.

### Modelo preditivo

- **Algoritmo:** Random Forest (200 árvores, `class_weight="balanced"`).
- **Validação:** divisão 80/20 estratificada (`random_state=42`) e validação cruzada de 5 partes.
- **Resultados:** acurácia de **99,90%** no teste e **99,74%** na validação cruzada; matriz de confusão com 734 verdadeiros negativos, 0 falsos positivos, 1 falso negativo e 265 verdadeiros positivos.
- **Modelo salvo:** `models/modelo_risco.joblib`; matriz de confusão em `models/matriz_confusao.png`.
- **Limitação importante:** o alvo foi gerado por regras fixas sobre as próprias variáveis, então a acurácia quase perfeita reflete a base simulada e não o desempenho em dados reais (detalhes no relatório técnico).

### Arquitetura multiagente

```
Dados do paciente
      │
      ▼
Orquestrador ──handoff──► Analista de Risco ──handoff──► Especialista em Protocolos ──handoff──► Planejador de Resposta
                           tool: prever_risco             tool: consultar_protocolos              tool: emitir_resposta_final
                           (modelo_risco.joblib)          (base de 10 protocolos)                 (Pydantic + guardrail)
```

| Agente | Papel |
|---|---|
| Orquestrador | Recebe os dados do paciente e coordena o fluxo por handoffs |
| Analista de Risco | Chama o modelo de ML e obtém probabilidade e classificação (BAIXO < 0,4 ≤ MODERADO < 0,7 ≤ ALTO) |
| Especialista em Protocolos | Identifica os protocolos acionados pelos dados e pelo risco |
| Planejador de Resposta | Consolida tudo e emite a resposta estruturada |

Recursos usados: **handoffs**, **tools**, **histórico de mensagens** (retomada do fluxo e pergunta de acompanhamento) e **validação de saída** (modelo Pydantic `RespostaFinal` + guardrail que confere coerência entre probabilidade, classificação e protocolos, com novas tentativas em caso de falha).

### Exemplo de execução

```
RISCO PREVISTO: 0.96
CLASSIFICAÇÃO: ALTO

PROTOCOLOS SUGERIDOS:
- Se carga do sistema > 85% → abrir unidades temporárias
- Se disponibilidade_recursos < 30% → redistribuir insumos
- Se densidade_eventos > 6 → ativar plano de contingência
- Se risco sistêmico alto → ativar estado de emergência
```

---

## 🗂 Estrutura do Projeto

```
.
├── agentes.py            # agentes, tools, handoffs, guardrail e execução do fluxo
├── dados.py              # geração da base sintética e novo caso de teste
├── train_model.py        # treino, avaliação e salvamento do modelo
├── modelo_service.py     # carrega o modelo salvo e faz a predição
├── protocolos.py         # base de protocolos e regras de acionamento
├── requirements.txt
├── models/
│   ├── modelo_risco.joblib
│   └── matriz_confusao.png
├── notebook/             # notebook do Google Colab (base, treino, métricas e simulação)
├── docs/
│   ├── relatorio_tecnico.pdf
│   └── arquitetura_multiagente.pdf
└── README.md
```

## ▶️ Como Executar

**Requisitos:** Python 3.10 ou superior.

```bash
# 1. instalar as dependências
pip install -r requirements.txt

# 2. treinar e salvar o modelo (gera models/modelo_risco.joblib)
python train_model.py

# 3. definir a chave do provedor de LLM (somente no terminal; nunca no código)
#    OpenAI (padrão, modelo gpt-4o-mini):
#    PowerShell:  $env:OPENAI_API_KEY="sua_chave"
#    Linux/macOS: export OPENAI_API_KEY="sua_chave"

# 4. executar o sistema multiagente
python agentes.py
```

**Usando outro provedor compatível com a API da OpenAI** (exemplo usado no desenvolvimento: Groq com `openai/gpt-oss-120b`):

```powershell
$env:LLM_BASE_URL="https://api.groq.com/openai/v1"
$env:LLM_API_KEY="sua_chave"
$env:LLM_MODEL="openai/gpt-oss-120b"
python agentes.py
```

## 📦 Dependências

`numpy`, `pandas`, `scikit-learn`, `joblib`, `matplotlib`, `openai-agents`, `pydantic` (veja `requirements.txt`).

## 🧠 Competências Desenvolvidas

- ✔️ Estruturar problemas de IA de ponta a ponta
- ✔️ Construção e avaliação de modelos supervisionados
- ✔️ Integração de modelos de ML a sistemas multiagente
- ✔️ Validação de saídas de LLM e tratamento de falhas
- ✔️ Documentação técnica clara
- ✔️ Versionamento e organização de código

## 📋 Licença

MODELO GIT FIAP por FIAP está licenciado sobre Attribution 4.0 International.
