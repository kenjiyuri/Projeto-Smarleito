"""Rotas de previsão (alta e demanda)."""

from fastapi import APIRouter

from app.schemas.leito import (
    PrevisaoAltaRequest,
    PrevisaoAltaResponse,
    PrevisaoDemandaRequest,
    PrevisaoDemandaResponse,
)
from app.services import previsao_service

router = APIRouter(prefix="/previsao", tags=["Previsão"])


@router.post("/alta", response_model=PrevisaoAltaResponse)
def prever_alta(req: PrevisaoAltaRequest):
    """
    Prevê a data de alta de um paciente com base em features clínicas.
    Usa Random Forest (se modelo treinado existir) ou heurística.
    """
    return previsao_service.prever_alta(req)


@router.post("/demanda", response_model=PrevisaoDemandaResponse)
def prever_demanda(req: PrevisaoDemandaRequest):
    """
    Prevê a demanda de leitos para os próximos N dias.
    Opcionalmente filtrado por setor.
    """
    return previsao_service.prever_demanda(req)
