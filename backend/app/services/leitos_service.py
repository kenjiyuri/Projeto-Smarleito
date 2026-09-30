"""Serviço de gestão de leitos (em memória para demo)."""

from datetime import date, datetime, timedelta
from typing import Optional
import random

from app.schemas.leito import (
    Leito,
    LeitoCreate,
    LeitoUpdate,
    StatusLeito,
    Setor,
    DashboardResumo,
)

# Banco em memória (substituir por PostgreSQL em produção)
_leitos: dict[int, Leito] = {}
_next_id = 1


def _seed():
    """Popula leitos de demonstração."""
    global _next_id
    if _leitos:
        return

    status_opts = list(StatusLeito)
    pesos = [0.22, 0.48, 0.12, 0.10, 0.08]
    setores = list(Setor)

    for i in range(1, 49):
        status = random.choices(status_opts, weights=pesos, k=1)[0]
        dias = random.randint(1, 18) if status == StatusLeito.OCUPADO else 0
        previsao = None
        if status == StatusLeito.OCUPADO:
            previsao = date.today() + timedelta(days=random.randint(0, 5))

        leito = Leito(
            id=_next_id,
            codigo=f"L-{i:03d}",
            setor=random.choice(setores),
            status=status,
            dias_internado=dias,
            previsao_alta=previsao,
            atualizado_em=datetime.utcnow(),
        )
        _leitos[_next_id] = leito
        _next_id += 1


def listar(
    setor: Optional[Setor] = None,
    status: Optional[StatusLeito] = None,
) -> list[Leito]:
    _seed()
    result = list(_leitos.values())
    if setor:
        result = [l for l in result if l.setor == setor]
    if status:
        result = [l for l in result if l.status == status]
    return sorted(result, key=lambda l: l.codigo)


def obter(leito_id: int) -> Optional[Leito]:
    _seed()
    return _leitos.get(leito_id)


def criar(dados: LeitoCreate) -> Leito:
    global _next_id
    _seed()
    leito = Leito(
        id=_next_id,
        codigo=dados.codigo,
        setor=dados.setor,
        status=dados.status,
        dias_internado=0,
        previsao_alta=None,
        atualizado_em=datetime.utcnow(),
    )
    _leitos[_next_id] = leito
    _next_id += 1
    return leito


def atualizar(leito_id: int, dados: LeitoUpdate) -> Optional[Leito]:
    leito = obter(leito_id)
    if not leito:
        return None
    update = dados.model_dump(exclude_unset=True)
    updated = leito.model_copy(update={**update, "atualizado_em": datetime.utcnow()})
    _leitos[leito_id] = updated
    return updated


def resumo() -> DashboardResumo:
    leitos = listar()
    total = len(leitos)
    contagem = {s: 0 for s in StatusLeito}
    for l in leitos:
        contagem[l.status] += 1

    altas = sum(
        1
        for l in leitos
        if l.previsao_alta and l.previsao_alta <= date.today() + timedelta(days=5)
    )

    ocupados = contagem[StatusLeito.OCUPADO]
    return DashboardResumo(
        total_leitos=total,
        livres=contagem[StatusLeito.LIVRE],
        ocupados=ocupados,
        em_higienizacao=contagem[StatusLeito.HIGIENIZACAO],
        reservados=contagem[StatusLeito.RESERVADO],
        interditados=contagem[StatusLeito.INTERDITADO],
        taxa_ocupacao=round(ocupados / total * 100, 1) if total else 0.0,
        altas_previstas_5d=altas,
    )
