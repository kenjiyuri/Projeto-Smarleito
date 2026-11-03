
"""
Pré-processamento compartilhado por treino, API e dashboard.

Esta é a única fonte da verdade para:
- A ordem das colunas (FEATURES)
- A codificação do setor (SETORES)

Qualquer mudança aqui exige retreinar o modelo.
"""

import numpy as np

SETORES = [
    "UTI",
    "Enfermaria",
    "Emergência",
    "Cirúrgico",
    "Pediatria",
]

FEATURES = [
    "idade",
    "dias_internado",
    "complexidade",
    "setor",
    "comorbidades",
]


def codificar_setor(setor) -> int:
    """Converte o nome do setor em índice."""

    nome = getattr(setor, "value", setor)

    if nome not in SETORES:
        raise ValueError(
            f"Setor desconhecido: {nome!r}. "
            f"Use um de {SETORES}"
        )

    return SETORES.index(nome)


def montar_features(
    idade,
    dias_internado,
    complexidade,
    setor,
    comorbidades
) -> np.ndarray:
    """Devolve a matriz de informações na ordem correta."""

    if isinstance(setor, str) or hasattr(setor, "value"):
        setores = [setor]
    else:
        setores = list(setor)
    colunas = [
        np.atleast_1d(idade),
        np.atleast_1d(dias_internado),
        np.atleast_1d(complexidade),
        np.array([
            codificar_setor(s)
            for s in setores
        ]),
        np.atleast_1d(comorbidades),
    ]

    return np.column_stack(colunas).astype(float)

