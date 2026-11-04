"""
Treina um Random Forest para prever dias restantes até a alta.

Gera dados sintéticos realistas e salva o modelo em ml/models/modelo_alta.joblib.

Uso:
    python ml/train_modelo_alta.py
"""
import sys
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1])
)

from ml.preprocessing import (
    SETORES,
    FEATURES,
    montar_features,
)
# ---------------------------------------------------------------------------
# Dados sintéticos
# Features: idade, dias_internado, complexidade (1-5), setor_encoded (0-4), comorbidades
# Target: dias_restantes até alta
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)
N = 10000
PERMANENCIA_MAX = 60
PISO_RESTANTES = 0.5

setor = RNG.integers(0, 5, N)  # 0=UTI … 4=Pediatria
eh_pediatria = setor == 4

# Pediatria: crianças e adolescentes; demais setores: adultos.
idade = np.where(
    eh_pediatria,
    RNG.integers(0, 18, N),
    RNG.integers(18, 95, N),
)

complexidade = RNG.integers(1, 6, N)

# Pediatria: até 2 comorbidades; demais setores: até 7.
comorbidades = np.where(
    eh_pediatria,
    RNG.integers(0, 3, N),
    RNG.integers(0, 8, N),
)

# Permanência base por setor
base_setor = np.array([7.5, 4.0, 1.5, 5.0, 3.5])[setor]

permanencia_esperada = (
    base_setor
    + np.maximum(0, (idade - 60) * 0.04)
    + (complexidade - 3) * 0.8
    + comorbidades * 0.5
    + RNG.normal(0, 1.2, N)
)

# Cauda longa: a maioria fica poucos dias, mas alguns ficam semanas ou meses
# Variação individual moderada, com teto de permanência.
permanencia_total = np.clip(
    np.clip(permanencia_esperada, 1, None) * RNG.lognormal(0, 0.35, N),
    1,
    PERMANENCIA_MAX,
)

# Dias já internado: sempre dentro da permanência total
dias_internado = np.floor(RNG.uniform(0, 1, N) * permanencia_total).astype(int)

# Dias que ainda faltam para a alta
dias_restantes = np.clip(
    permanencia_total - dias_internado,
    PISO_RESTANTES,
    PERMANENCIA_MAX,
)


def validar_dados_sinteticos(idade, dias_internado, dias_restantes, setor):
    """Falha cedo se o gerador produzir dados degenerados (issue #7)."""
    problemas = []

    if dias_internado.max() > PERMANENCIA_MAX:
        problemas.append(
            f"dias_internado passa de {PERMANENCIA_MAX} "
            f"(max={dias_internado.max()})"
        )

    if (dias_restantes <= PISO_RESTANTES).mean() > 0.10:
        problemas.append(
            "mais de 10% dos alvos presos no piso de dias restantes"
        )

    if ((setor == 4) & (idade >= 18)).any() or (
        (setor != 4) & (idade < 18)
    ).any():
        problemas.append(
            "combinação impossível de idade e setor (Pediatria x adulto)"
        )

    for i, nome in enumerate(SETORES):
        if dias_restantes[setor == i].std() < 0.5:
            problemas.append(f"alvo quase constante no setor {nome}")

    for nome, col in [
        ("idade", idade),
        ("dias_internado", dias_internado),
        ("dias_restantes", dias_restantes),
    ]:
        if np.unique(col).size < 5:
            problemas.append(f"feature {nome} com poucos valores distintos")

    if problemas:
        raise ValueError(
            "Dados sintéticos degenerados: " + "; ".join(problemas)
        )


validar_dados_sinteticos(
    idade,
    dias_internado,
    dias_restantes,
    setor,
)

print("Dados sintéticos OK. Média de dias restantes por setor:")

for i, nome in enumerate(SETORES):
    mascara = setor == i
    print(
        f"  {nome:<11} n={int(mascara.sum()):>5}  "
        f"média={dias_restantes[mascara].mean():.1f}"
    )
setor_nome = [SETORES[i] for i in setor]

X = montar_features(
    idade,
    dias_internado,
    complexidade,
    setor_nome,
    comorbidades
)

assert X.shape[1] == len(FEATURES)
y = dias_restantes

print(f"dias_internado: min={dias_internado.min()}, max={dias_internado.max()}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(
    n_estimators=120,
    max_depth=10,
    min_samples_leaf=4,
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train)

pred = model.predict(X_test)
mae = mean_absolute_error(y_test, pred)
r2 = r2_score(y_test, pred)

baseline = mean_absolute_error(
    y_test,
    np.full_like(y_test, y_train.mean()),
)

print(
    f"MAE: {mae:.2f} dias "
    f"(baseline 'sempre a média': {baseline:.2f})"
)
print(f"R²:  {r2:.3f}")

out_dir = Path(__file__).parent / "models"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "modelo_alta.joblib"
joblib.dump(model, out_path)
print(f"Modelo salvo em: {out_path}")


