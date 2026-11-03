"""
Serviço de previsão com Random Forest / lógica heurística.

Em produção: carregar modelo treinado (joblib) de ml/models/.
Aqui usamos um modelo simplificado + heurísticas para funcionar offline.
"""

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
)
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

def prever_demanda(req: PrevisaoDemandaRequest) -> PrevisaoDemandaResponse:
    """
    Gera série de demanda prevista para os próximos N dias.
    Usa sazonalidade senoidal + ruído controlado (demo).
    """
    hoje = date.today()
    previsoes = []

    # Baseline por setor
    baseline = {
        Setor.UTI: 12,
        Setor.ENFERMARIA: 28,
        Setor.EMERGENCIA: 18,
        Setor.CIRURGICO: 15,
        Setor.PEDIATRIA: 10,
    }

    setores = [req.setor] if req.setor else list(Setor)

    for i in range(req.dias_a_frente):
        d = hoje + timedelta(days=i)
        # Sazonalidade semanal (fim de semana ↓) + tendência leve
        fator_semana = 0.85 if d.weekday() >= 5 else 1.0
        onda = 1 + 0.12 * math.sin(2 * math.pi * i / 7)

        if req.setor:
            dem = int(baseline[req.setor] * fator_semana * onda)
            livres_est = max(2, int(dem * 0.25))
        else:
            dem = int(sum(baseline.values()) * fator_semana * onda / len(Setor) * 1.1)
            livres_est = max(5, int(dem * 0.22))

        previsoes.append(
            PrevisaoDemandaItem(
                data=d,
                demanda_prevista=dem,
                leitos_livres_estimados=livres_est,
            )
        )

    return PrevisaoDemandaResponse(
        setor=req.setor.value if req.setor else "Todos",
        previsoes=previsoes,
    )
