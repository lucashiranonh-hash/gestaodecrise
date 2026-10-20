"""Parte 2: sistema multiagente (OpenAI Agents SDK).

Fluxo: Orquestrador -> Analista de Risco -> Especialista em Protocolos -> Planejador
(handoffs). O Planejador devolve saída estruturada validada (Pydantic + guardrail).
Requer: export OPENAI_API_KEY=...
"""
import asyncio
import json
import os
from datetime import datetime

from openai import AsyncOpenAI, BadRequestError
from agents import (Agent, OutputGuardrailTripwireTriggered, ModelSettings, handoff, OpenAIChatCompletionsModel, set_tracing_disabled, GuardrailFunctionOutput, Runner, function_tool,
                    output_guardrail, RunContextWrapper)
from pydantic import BaseModel, Field

from dados import NOVO_CASO
from modelo_service import predizer
from protocolos import classificar_risco, protocolos_acionados

# --- Configuração do LLM ---
# Padrão: OpenAI (gpt-4o-mini). Para usar Llama (Groq/OpenRouter/Together), defina no terminal:
#   LLM_BASE_URL, LLM_API_KEY e LLM_MODEL (veja o README).
if os.getenv("LLM_BASE_URL"):
    set_tracing_disabled(True)  # tracing do SDK exige chave da OpenAI
    MODELO_LLM = OpenAIChatCompletionsModel(
        model=os.environ["LLM_MODEL"],
        openai_client=AsyncOpenAI(base_url=os.environ["LLM_BASE_URL"],
                                  api_key=os.environ["LLM_API_KEY"]))
else:
    MODELO_LLM = "gpt-4o-mini"


# ---------- HANDOFFS ----------
# Esquema não vazio: alguns provedores (ex.: Groq) rejeitam handoff sem "properties".
class MotivoHandoff(BaseModel):
    motivo: str = Field(description="Breve motivo do encaminhamento")


async def _ao_encaminhar(ctx: RunContextWrapper, dados: MotivoHandoff):
    print(f"[handoff] {dados.motivo}")  # rastreia o fluxo entre agentes


def passar_para(agente: Agent):
    return handoff(agente, input_type=MotivoHandoff, on_handoff=_ao_encaminhar)


# ---------- TOOLS ----------
@function_tool
def prever_risco(idade_media: float, carga_sistema: float, tempo_espera: float,
                 fc_media: float, spo2_media: float, pa_sistolica: float,
                 temperatura: float, densidade_eventos: int, eventos_24h: int,
                 disponibilidade_recursos: float) -> str:
    """Calcula a probabilidade de pico de risco cardíaco com o modelo de ML treinado."""
    dados = {k: v for k, v in locals().items()}
    prob = predizer(dados)
    return json.dumps({"probabilidade": round(prob, 4),
                       "classificacao": classificar_risco(prob), "dados": dados})


@function_tool
def consultar_protocolos(probabilidade: float, spo2_media: float, carga_sistema: float,
                         fc_media: float, eventos_24h: int, disponibilidade_recursos: float,
                         tempo_espera: float, temperatura: float, densidade_eventos: int,
                         pa_sistolica: float) -> str:
    """Consulta a base de protocolos e devolve os acionados pelos dados do paciente."""
    prob = probabilidade
    dados = {k: v for k, v in locals().items() if k != "probabilidade"}
    return json.dumps(protocolos_acionados(dados, prob), ensure_ascii=False)


# ---------- SAÍDA ESTRUTURADA ----------
class RespostaFinal(BaseModel):
    risco_previsto: float = Field(description="Probabilidade entre 0 e 1 retornada pela tool")
    classificacao_risco: str = Field(description="BAIXO, MODERADO ou ALTO")
    protocolos_sugeridos: list[str]
    acoes_operacionais: list[str] = Field(description="Ações práticas priorizadas")


@function_tool
def emitir_resposta_final(risco_previsto: float, classificacao_risco: str,
                          protocolos_sugeridos: list[str], acoes_operacionais: list[str]) -> str:
    """Emite a recomendação final estruturada. Valida o formato com Pydantic."""
    return RespostaFinal(risco_previsto=risco_previsto, classificacao_risco=classificacao_risco,
                         protocolos_sugeridos=protocolos_sugeridos,
                         acoes_operacionais=acoes_operacionais).model_dump_json()


@output_guardrail
async def validar_saida(ctx: RunContextWrapper, agent: Agent, saida):
    """Validação de saída: formato válido, probabilidade coerente com a classificação
    e protocolos preenchidos em risco alto."""
    try:
        r = RespostaFinal.model_validate_json(str(saida))
    except Exception:
        return GuardrailFunctionOutput(output_info="saída fora do formato", tripwire_triggered=True)
    p, c = r.risco_previsto, r.classificacao_risco.upper()
    incoerente = (not 0 <= p <= 1) or c != classificar_risco(p) \
        or (c == "ALTO" and not r.protocolos_sugeridos)
    return GuardrailFunctionOutput(output_info={"prob": p, "classe": c},
                                   tripwire_triggered=incoerente)


# ---------- AGENTES ----------
# O Planejador finaliza chamando a tool emitir_resposta_final (stop_on_first_tool):
# funciona melhor que "JSON schema" em provedores alternativos.
planejador = Agent(
    name="Planejador de Resposta", model=MODELO_LLM, tools=[emitir_resposta_final],
    tool_use_behavior="stop_on_first_tool", output_guardrails=[validar_saida],
    instructions=("Você é o Planejador de Resposta. Consolide o risco previsto (valor EXATO "
                  "retornado por prever_risco), a classificação e os protocolos retornados por "
                  "consultar_protocolos (não invente protocolos). Sugira 3 a 5 ações operacionais "
                  "curtas e priorizadas. Sua ÚNICA ação é chamar a tool emitir_resposta_final. "
                  "NUNCA escreva texto nem mensagens de transferência."))

especialista_protocolos = Agent(
    name="Especialista em Protocolos", model=MODELO_LLM, tools=[consultar_protocolos],
    handoffs=[passar_para(planejador)],
    instructions=("Você é o Especialista em Protocolos. Passos obrigatórios, nesta ordem: "
                  "(1) chame a tool consultar_protocolos com a probabilidade e os dados do "
                  "paciente que estão no histórico; (2) logo após receber o resultado, chame "
                  "transfer_to_planejador_de_resposta. NUNCA escreva a resposta final nem mensagens de "
                  "transferência: responda apenas chamando tools."))

analista_risco = Agent(
    name="Analista de Risco", model=MODELO_LLM, tools=[prever_risco],
    handoffs=[passar_para(especialista_protocolos)],
    model_settings=ModelSettings(tool_choice="required"),  # obriga a chamar o modelo de ML
    instructions=("Você é o Analista de Risco. Passos obrigatórios, nesta ordem: "
                  "(1) extraia os 10 dados clínicos do paciente e chame a tool prever_risco; "
                  "(2) logo após receber o resultado, chame transfer_to_especialista_em_protocolos. "
                  "NUNCA responda em texto: sua resposta é sempre a chamada de uma tool."))

orquestrador = Agent(
    name="Orquestrador", model=MODELO_LLM, handoffs=[passar_para(analista_risco)],
    instructions=("Você coordena o pipeline do CardioIA. Ao receber dados de um paciente, faça "
                  "handoff imediato ao Analista de Risco. Não responda você mesmo."))


def formatar(r: RespostaFinal) -> str:
    linhas = [f"RISCO PREVISTO: {r.risco_previsto:.2f}",
              f"CLASSIFICAÇÃO: {r.classificacao_risco}", "", "PROTOCOLOS SUGERIDOS:"]
    linhas += [f"- {p}" for p in r.protocolos_sugeridos]
    linhas += ["", "AÇÕES OPERACIONAIS:"] + [f"- {a}" for a in r.acoes_operacionais]
    return "\n".join(linhas)


def extrair_resposta(res) -> RespostaFinal | None:
    """Converte a saída final em RespostaFinal (validação Pydantic); None se inválida."""
    try:
        return RespostaFinal.model_validate_json(str(res.final_output))
    except Exception:
        return None


async def executar_fluxo(historico):
    """Roda o pipeline; se o modelo parar no meio ou gerar saída inválida, retoma/tenta de novo."""
    for tentativa in range(1, 4):
        try:
            res = await Runner.run(orquestrador, historico, max_turns=12)
            for _ in range(3):  # retomada a partir do último agente
                if extrair_resposta(res):
                    return res
                print(f"[retomada] fluxo parou em '{res.last_agent.name}', continuando...")
                res = await Runner.run(
                    res.last_agent,
                    res.to_input_list() + [{"role": "user", "content":
                        "Continue o fluxo: execute seus passos obrigatórios (tool e handoff)."}],
                    max_turns=12)
            if extrair_resposta(res):
                return res
        except (BadRequestError, OutputGuardrailTripwireTriggered) as e:
            print(f"[tentativa {tentativa}/3] falha do modelo/validação: {type(e).__name__}")
    return None


async def main():
    historico = [{"role": "user", "content": "Novo paciente: " + json.dumps(NOVO_CASO)}]
    res = await executar_fluxo(historico)
    if res is None:
        print("Não foi possível concluir o fluxo. Rode novamente.")
        return
    print("\n" + formatar(extrair_resposta(res)))
    print("\nAgente final:", res.last_agent.name)

    # Histórico de mensagens: a segunda interação reaproveita a conversa anterior
    # (dados do paciente + resposta do pipeline) para um agente de esclarecimento sem tools.
    esclarecedor = Agent(name="Assistente de Esclarecimento", model=MODELO_LLM,
                         instructions="Responda em português, de forma curta, usando só o contexto.")
    try:
        historico2 = historico + [
            {"role": "assistant", "content": formatar(extrair_resposta(res))},
            {"role": "user", "content": "Qual desses protocolos é o mais urgente e por quê?"}]
        res2 = await Runner.run(esclarecedor, historico2)
        print("\nFollow-up:", res2.final_output)
    except Exception as e:
        print("\n(follow-up ignorado:", type(e).__name__, ")")


if __name__ == "__main__":
    asyncio.run(main())
