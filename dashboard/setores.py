"""
Indicadores e regras de acompanhamento específicos de cada setor do hospital.

ATENÇÃO: os parâmetros de PERFIS (permanência esperada e limites de ocupação) são
valores de referência para demonstração. Ajuste-os conforme os protocolos reais
da instituição. Os cálculos não dependem do Streamlit.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

# Parâmetros de referência por setor (editáveis)
#   esperado : permanência esperada, em dias
#   atencao  : ocupação (%) a partir da qual o setor entra em "Atenção"
#   critico  : ocupação (%) a partir da qual o setor entra em "Crítico"
PERFIS = {
    "UTI": {
        "descricao": "Pacientes graves, com monitorização contínua e necessidade de reserva para urgências.",
        "foco": "Reserva para urgências e permanência prolongada",
        "esperado": 7.5, "atencao": 80, "critico": 90,
    },
    "Emergência": {
        "descricao": "Porta de entrada do hospital: avaliação, estabilização e encaminhamento.",
        "foco": "Tempo de permanência e fluxo de novos atendimentos",
        "esperado": 1.5, "atencao": 75, "critico": 90,
    },
    "Enfermaria": {
        "descricao": "Internação clínica de pacientes estáveis, que recebem transferências da UTI e da Emergência.",
        "foco": "Giro de leitos e capacidade para receber transferências",
        "esperado": 4.0, "atencao": 85, "critico": 95,
    },
    "Cirúrgico": {
        "descricao": "Pré e pós-operatório, com agenda cirúrgica que depende de leitos disponíveis.",
        "foco": "Disponibilidade de leitos para cirurgias eletivas",
        "esperado": 5.0, "atencao": 85, "critico": 95,
    },
    "Pediatria": {
        "descricao": "Atendimento de crianças e adolescentes, com acompanhamento familiar.",
        "foco": "Disponibilidade de leitos e permanência pediátrica",
        "esperado": 3.5, "atencao": 80, "critico": 90,
    },
}

COLUNAS_LEITOS = ["codigo", "setor", "status", "dias_internado", "previsao_alta"]


# ---------------------------------------------------------------------------
# Preparação dos dados
# ---------------------------------------------------------------------------
def normalizar_leitos(df: pd.DataFrame) -> pd.DataFrame:
    """Padroniza nomes de colunas e tipos, venham os dados da API ou do CSV."""
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    for col in COLUNAS_LEITOS:
        if col not in df.columns:
            df[col] = None
    df["dias_internado"] = pd.to_numeric(df["dias_internado"], errors="coerce").fillna(0)
    df["previsao_alta"] = pd.to_datetime(df["previsao_alta"], errors="coerce")
    return df[COLUNAS_LEITOS]


def data_referencia(df: pd.DataFrame, ultima_data_historico: date | None = None,
                    hoje: date | None = None) -> date:
    """Data a partir da qual as previsões de alta devem ser contadas.

    Dados ao vivo (API) têm previsões a partir de hoje. Já o CSV de demonstração
    guarda datas do dia em que foi gerado; nesse caso usamos a última data do
    histórico de demanda, que foi gerada junto com ele.
    """
    hoje = hoje or date.today()
    previsoes = df["previsao_alta"].dropna()
    if previsoes.empty or previsoes.max().date() >= hoje:
        return hoje
    if ultima_data_historico and previsoes.max().date() >= ultima_data_historico:
        return ultima_data_historico
    return previsoes.min().date()


# ---------------------------------------------------------------------------
# Indicadores
# ---------------------------------------------------------------------------
def calcular(df_setor: pd.DataFrame, setor: str, ref: date) -> dict:
    """Indicadores de um setor. Espera o resultado de normalizar_leitos já filtrado."""
    perfil = PERFIS[setor]
    total = len(df_setor)
    contagem = df_setor["status"].value_counts()
    ocupados_df = df_setor[df_setor["status"] == "Ocupado"].copy()
    ocupados_df["dias_ate_alta"] = (
        ocupados_df["previsao_alta"].dt.normalize() - pd.Timestamp(ref)
    ).dt.days

    m = {
        "total": total,
        "ocupados": int(contagem.get("Ocupado", 0)),
        "livres": int(contagem.get("Livre", 0)),
        "higienizacao": int(contagem.get("Em higienização", 0)),
        "reservados": int(contagem.get("Reservado", 0)),
        "interditados": int(contagem.get("Interditado", 0)),
    }
    m["ocupacao"] = round(m["ocupados"] / total * 100, 1) if total else 0.0
    m["permanencia_media"] = (
        round(float(ocupados_df["dias_internado"].mean()), 1) if m["ocupados"] else None
    )
    m["acima_esperado"] = int((ocupados_df["dias_internado"] > perfil["esperado"]).sum())
    m["altas_48h"] = int((ocupados_df["dias_ate_alta"] <= 2).sum())
    m["altas_vencidas"] = int((ocupados_df["dias_ate_alta"] < 0).sum())
    m["disponiveis"] = m["livres"] + m["higienizacao"]  # livres + em preparo
    m["disponibilidade_48h"] = m["disponiveis"] + m["altas_48h"]
    m["_ocupados_df"] = ocupados_df
    return m


def situacao(ocupacao: float, setor: str) -> tuple[str, str, str]:
    """Devolve (nível, chave de cor, mensagem) conforme a ocupação do setor."""
    p = PERFIS[setor]
    if ocupacao >= p["critico"]:
        return ("Crítico", "Ocupado",
                f"Ocupação de {fmt(ocupacao)}%, acima do limite crítico de {p['critico']}%.")
    if ocupacao >= p["atencao"]:
        return ("Atenção", "Em higienização",
                f"Ocupação de {fmt(ocupacao)}%, acima do limite de atenção de {p['atencao']}%.")
    return ("Normal", "Livre",
            f"Ocupação de {fmt(ocupacao)}%, dentro do limite de atenção de {p['atencao']}%.")


def kpis(setor: str, m: dict) -> list[tuple[str, str]]:
    """Quatro indicadores principais, diferentes para cada setor."""
    p = PERFIS[setor]
    esp = fmt(p["esperado"])
    perm = f"{fmt(m['permanencia_media'])} dias" if m["permanencia_media"] is not None else "—"
    ocup = f"{fmt(m['ocupacao'])}%"
    if setor == "UTI":
        return [("Ocupação", ocup), ("Reserva para urgências", str(m["livres"])),
                ("Permanência média", perm), (f"Acima de {esp} dias", str(m["acima_esperado"]))]
    if setor == "Emergência":
        return [("Ocupação", ocup), ("Prontos para novos atendimentos", str(m["disponiveis"])),
                (f"Acima de {esp} dias", str(m["acima_esperado"])), ("Altas em 48 h", str(m["altas_48h"]))]
    if setor == "Enfermaria":
        return [("Ocupação", ocup), ("Aptos a receber transferências", str(m["disponiveis"])),
                ("Permanência média", perm), ("Altas em 48 h", str(m["altas_48h"]))]
    if setor == "Cirúrgico":
        return [("Ocupação", ocup), ("Disponíveis para eletivas", str(m["disponiveis"])),
                ("Disponibilidade em 48 h", str(m["disponibilidade_48h"])), ("Permanência média", perm)]
    return [("Ocupação", ocup), ("Leitos livres", str(m["livres"])),
            ("Permanência média", perm), ("Altas em 48 h", str(m["altas_48h"]))]


def regras(setor: str, m: dict) -> list[dict]:
    """Pontos de acompanhamento do setor. Cada item: título, ok (bool) e detalhe."""
    p = PERFIS[setor]
    esp = fmt(p["esperado"])
    oc = m["ocupados"]

    def item(titulo, ok, detalhe_ok, detalhe_alerta):
        return {"titulo": titulo, "ok": ok, "detalhe": detalhe_ok if ok else detalhe_alerta}

    prolongada = item(
        f"Permanência acima de {esp} dias", m["acima_esperado"] == 0,
        "Nenhum paciente acima do esperado.",
        f"{m['acima_esperado']} de {oc} paciente(s) acima do esperado.",
    )
    interdicao = item(
        "Leitos interditados", m["interditados"] == 0,
        "Nenhum leito interditado.", f"{m['interditados']} leito(s) fora de uso.",
    )
    vencidas = item(
        "Altas previstas em atraso", m["altas_vencidas"] == 0,
        "Nenhuma alta em atraso.", f"{m['altas_vencidas']} alta(s) prevista(s) já vencida(s).",
    )

    if setor == "UTI":
        return [
            item("Reserva para urgências", m["livres"] >= 1,
                 f"{m['livres']} leito(s) livre(s) para urgências.",
                 "Nenhum leito livre para urgências."),
            prolongada, vencidas, interdicao,
        ]
    if setor == "Emergência":
        limite = max(1, round(oc * 0.2)) if oc else 0
        return [
            item("Leitos para novos atendimentos", m["disponiveis"] >= 1,
                 f"{m['disponiveis']} leito(s) livre(s) ou em preparo.",
                 "Nenhum leito livre ou em preparo."),
            item(f"Permanência acima de {esp} dias", m["acima_esperado"] <= limite,
                 f"{m['acima_esperado']} de {oc} paciente(s), dentro do tolerado (até 20%).",
                 f"{m['acima_esperado']} de {oc} paciente(s), acima do tolerado (20%)."),
            vencidas, interdicao,
        ]
    if setor == "Enfermaria":
        return [
            item("Capacidade para transferências", m["disponiveis"] >= 2,
                 f"{m['disponiveis']} leito(s) livre(s) ou em preparo.",
                 f"Apenas {m['disponiveis']} leito(s) livre(s) ou em preparo (mínimo de 2)."),
            item("Fluxo de altas", m["altas_48h"] >= 1,
                 f"{m['altas_48h']} alta(s) prevista(s) em 48 h.",
                 "Nenhuma alta prevista em 48 h."),
            prolongada, vencidas,
        ]
    if setor == "Cirúrgico":
        return [
            item("Leitos para cirurgias eletivas", m["disponiveis"] >= 1,
                 f"{m['disponiveis']} leito(s) livre(s) ou em preparo.",
                 "Nenhum leito disponível para eletivas."),
            item("Disponibilidade projetada em 48 h", m["disponibilidade_48h"] >= 2,
                 f"{m['disponibilidade_48h']} leito(s) previsto(s) para 48 h.",
                 f"Apenas {m['disponibilidade_48h']} leito(s) previsto(s) para 48 h (mínimo de 2)."),
            prolongada, interdicao,
        ]
    return [  # Pediatria
        item("Leito disponível", m["livres"] >= 1,
             f"{m['livres']} leito(s) livre(s).", "Nenhum leito livre."),
        prolongada, vencidas, interdicao,
    ]


def projecao_altas(m: dict, ref: date, dias: int = 7) -> pd.DataFrame:
    """Altas previstas por dia e disponibilidade acumulada nos próximos dias."""
    oc = m["_ocupados_df"]
    linhas, acumulado = [], m["disponiveis"]
    for d in range(dias):
        # altas vencidas contam no primeiro dia
        altas = int((oc["dias_ate_alta"] <= 0).sum()) if d == 0 else int((oc["dias_ate_alta"] == d).sum())
        acumulado += altas
        data = pd.Timestamp(ref) + pd.Timedelta(days=d)
        linhas.append({"data": data.strftime("%d/%m"), "altas": altas, "disponiveis": acumulado})
    return pd.DataFrame(linhas)


def fmt(valor, casas: int = 1) -> str:
    """Número no padrão brasileiro (vírgula decimal); sem casas se for inteiro."""
    if valor is None:
        return "—"
    if float(valor).is_integer():
        return str(int(valor))
    return f"{valor:.{casas}f}".replace(".", ",")