"""
Serviço de previsão com Random Forest / lógica heurística.

Em produção: carregar modelo treinado (joblib) de ml/models/.
Aqui usamos um modelo simplificado + heurísticas para funcionar offline.
"""

import csv
import math
import sys
import threading

from datetime import date, timedelta
from pathlib import Path
from typing import Optional

from app.schemas.leito import (
    PrevisaoAltaRequest,
    PrevisaoAltaResponse,
    PrevisaoDemandaRequest,
    PrevisaoDemandaResponse,
    PrevisaoDemandaItem,
    Setor,
    StatusLeito,
)
from app.services import leitos_service
# Caminho raiz do projeto
_RAIZ = Path(__file__).resolve().parents[3]

# Permite importar o pré-processamento compartilhado em ml/
if str(_RAIZ) not in sys.path:
    sys.path.insert(0, str(_RAIZ))

from ml.preprocessing import montar_features

# Caminho do modelo treinado
_MODEL_PATH = _RAIZ / "ml" / "models" / "modelo_alta.joblib"

_model = None
_model_mtime = None
_lock = threading.Lock()


def _get_model():
    """
    Carrega o modelo quando necessário e recarrega se o arquivo mudar.
    """
    global _model, _model_mtime

    try:
        mtime = _MODEL_PATH.stat().st_mtime
    except OSError:
        return None

    with _lock:
        if _model is None or mtime != _model_mtime:
            try:
                import joblib

                _model = joblib.load(_MODEL_PATH)
                _model_mtime = mtime
            except Exception:
                return _model

        return _model


# Fatores médios de permanência por setor (dias)
_PERMANENCIA_SETOR = {
    Setor.UTI: 7.5,
    Setor.ENFERMARIA: 4.0,
    Setor.EMERGENCIA: 1.5,
    Setor.CIRURGICO: 5.0,
    Setor.PEDIATRIA: 3.5,
}

def prever_alta(req: PrevisaoAltaRequest) -> PrevisaoAltaResponse:
    """
    Prevê quantos dias restam até a alta.

    Features: idade, dias_internado, complexidade, setor, comorbidades.
    """
    modelo = _get_model()

    if modelo is not None:
        X = montar_features(
            req.idade,
            req.dias_internado,
            req.complexidade,
            req.setor,
            req.comorbidades,
        )

        dias = float(modelo.predict(X)[0])
        confianca = 0.82
    else:
        # Heurística: permanência esperada − dias já internado + ajustes
        base = _PERMANENCIA_SETOR.get(req.setor, 4.0)
        ajuste_idade = max(0, (req.idade - 60) * 0.04)
        ajuste_complex = (req.complexidade - 3) * 0.8
        ajuste_comorb = req.comorbidades * 0.5

        permanencia_esperada = (
            base + ajuste_idade + ajuste_complex + ajuste_comorb
        )

        dias = max(0.5, permanencia_esperada - req.dias_internado)
        confianca = 0.65

    dias = round(dias, 1)
    data_prevista = date.today() + timedelta(days=math.ceil(dias))

    return PrevisaoAltaResponse(
        dias_restantes=dias,
        data_prevista=data_prevista,
        confianca=confianca,
    )

# ---------------------------------------------------------------------------
# Previsão de demanda (issue #15)
# Antes: baselines fixos por setor + senoide com fase presa ao "dia de hoje".
# Agora: ponto de partida = ocupação REAL atual (leitos) e padrão semanal
# estimado do histórico (data/demanda_historico.csv), sem funções inventadas.
# ---------------------------------------------------------------------------
_HISTORICO_PATH = _RAIZ / "data" / "demanda_historico.csv"
_FATORES_PADRAO = [1.0, 1.0, 1.0, 1.0, 1.0, 0.85, 0.85]  # seg..dom, usado sem histórico
_cache_fatores = {"mtime": None, "valores": _FATORES_PADRAO}


def _fatores_semanais() -> list[float]:
    """
    Fator multiplicativo por dia da semana (0=segunda), estimado do histórico.
    Suavizado pela metade em direção a 1.0 (histórico curto = pouca confiança).
    Sem histórico suficiente (menos de 14 dias, ou menos de 3 por dia da semana),
    usa o padrão. Recarrega sozinho quando o CSV muda.
    """
    try:
        mtime = _HISTORICO_PATH.stat().st_mtime
    except OSError:
        return _FATORES_PADRAO
    if _cache_fatores["mtime"] == mtime:
        return _cache_fatores["valores"]

    por_dia: dict[int, list[float]] = {i: [] for i in range(7)}
    try:
        with open(_HISTORICO_PATH, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                d = date.fromisoformat(row["data"])
                por_dia[d.weekday()].append(float(row["ocupacao_real"]))
    except (OSError, KeyError, ValueError):
        return _FATORES_PADRAO

    todos = [v for vs in por_dia.values() for v in vs]
    if len(todos) < 14 or any(len(vs) < 3 for vs in por_dia.values()):
        valores = _FATORES_PADRAO
    else:
        media = sum(todos) / len(todos)
        valores = [1 + 0.5 * (sum(vs) / len(vs) / media - 1) for vs in por_dia.values()]

    _cache_fatores["mtime"], _cache_fatores["valores"] = mtime, valores
    return valores


def prever_demanda(req: PrevisaoDemandaRequest) -> PrevisaoDemandaResponse:
    """
    Prevê a demanda de leitos para os próximos N dias.

    demanda(dia) = leitos em uso hoje (ocupados + reservados)
                   x fator do dia da semana / fator de hoje
    Hoje coincide com o valor real; os demais dias seguem o padrão semanal.
    Livres estimados = capacidade (leitos não interditados) - demanda.
    """
    hoje = date.today()
    fatores = _fatores_semanais()
    fator_hoje = fatores[hoje.weekday()]

    leitos = leitos_service.listar(setor=req.setor)
    capacidade = sum(1 for l in leitos if l.status != StatusLeito.INTERDITADO)
    em_uso = sum(1 for l in leitos if l.status in (StatusLeito.OCUPADO, StatusLeito.RESERVADO))

    previsoes = []
    for i in range(req.dias_a_frente):
        d = hoje + timedelta(days=i)
        dem = round(em_uso * fatores[d.weekday()] / fator_hoje)
        dem = max(0, min(dem, capacidade))
        previsoes.append(
            PrevisaoDemandaItem(
                data=d,
                demanda_prevista=dem,
                leitos_livres_estimados=max(0, capacidade - dem),
            )
        )

    return PrevisaoDemandaResponse(
        setor=req.setor.value if req.setor else "Todos",
        previsoes=previsoes,
    )