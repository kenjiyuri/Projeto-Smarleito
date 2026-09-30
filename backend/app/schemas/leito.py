"""Schemas Pydantic para leitos e previsões."""

from datetime import date, datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class StatusLeito(str, Enum):
    LIVRE = "Livre"
    OCUPADO = "Ocupado"
    HIGIENIZACAO = "Em higienização"
    RESERVADO = "Reservado"
    INTERDITADO = "Interditado"


class Setor(str, Enum):
    UTI = "UTI"
    ENFERMARIA = "Enfermaria"
    EMERGENCIA = "Emergência"
    CIRURGICO = "Cirúrgico"
    PEDIATRIA = "Pediatria"


class LeitoBase(BaseModel):
    codigo: str = Field(..., example="L-001")
    setor: Setor
    status: StatusLeito


class LeitoCreate(LeitoBase):
    pass


class LeitoUpdate(BaseModel):
    status: Optional[StatusLeito] = None
    setor: Optional[Setor] = None


class Leito(LeitoBase):
    id: int
    dias_internado: int = 0
    previsao_alta: Optional[date] = None
    atualizado_em: datetime

    class Config:
        from_attributes = True


class PrevisaoAltaRequest(BaseModel):
    """Dados clínicos para prever data de alta."""
    idade: int = Field(..., ge=0, le=120)
    dias_internado: int = Field(..., ge=0)
    complexidade: int = Field(..., ge=1, le=5, description="1=baixa … 5=alta")
    setor: Setor
    comorbidades: int = Field(0, ge=0, le=10)


class PrevisaoAltaResponse(BaseModel):
    dias_restantes: float
    data_prevista: date
    confianca: float = Field(..., ge=0, le=1)


class PrevisaoDemandaRequest(BaseModel):
    setor: Optional[Setor] = None
    dias_a_frente: int = Field(7, ge=1, le=30)


class PrevisaoDemandaItem(BaseModel):
    data: date
    demanda_prevista: int
    leitos_livres_estimados: int


class PrevisaoDemandaResponse(BaseModel):
    setor: Optional[str]
    previsoes: list[PrevisaoDemandaItem]


class DashboardResumo(BaseModel):
    total_leitos: int
    livres: int
    ocupados: int
    em_higienizacao: int
    reservados: int
    interditados: int
    taxa_ocupacao: float
    altas_previstas_5d: int
