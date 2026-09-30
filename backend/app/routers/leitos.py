"""Rotas de gestão de leitos."""

from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from app.schemas.leito import (
    Leito,
    LeitoCreate,
    LeitoUpdate,
    StatusLeito,
    Setor,
    DashboardResumo,
)
from app.services import leitos_service

router = APIRouter(prefix="/leitos", tags=["Leitos"])


@router.get("", response_model=list[Leito])
def listar_leitos(
    setor: Optional[Setor] = Query(None),
    status: Optional[StatusLeito] = Query(None),
):
    """Lista leitos com filtros opcionais de setor e status."""
    return leitos_service.listar(setor=setor, status=status)


@router.get("/resumo", response_model=DashboardResumo)
def resumo_dashboard():
    """KPIs agregados para o dashboard."""
    return leitos_service.resumo()


@router.get("/{leito_id}", response_model=Leito)
def obter_leito(leito_id: int):
    leito = leitos_service.obter(leito_id)
    if not leito:
        raise HTTPException(status_code=404, detail="Leito não encontrado")
    return leito


@router.post("", response_model=Leito, status_code=201)
def criar_leito(dados: LeitoCreate):
    return leitos_service.criar(dados)


@router.patch("/{leito_id}", response_model=Leito)
def atualizar_leito(leito_id: int, dados: LeitoUpdate):
    leito = leitos_service.atualizar(leito_id, dados)
    if not leito:
        raise HTTPException(status_code=404, detail="Leito não encontrado")
    return leito
