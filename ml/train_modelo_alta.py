"""
Treina um Random Forest para prever dias restantes até a alta.

Gera dados sintéticos realistas e salva o modelo em ml/models/modelo_alta.joblib.

Uso:
    python ml/train_modelo_alta.py
"""
import json
import sys
from datetime import datetime
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import clone
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_val_score, train_test_split
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

# ---------------------------------------------------------------------------
# Validação e seleção de hiperparâmetros (issue #16)
#   1. Separa 20% como TESTE FINAL, que não participa de nenhuma escolha.
#   2. Escolhe os hiperparâmetros por validação cruzada (5 partes) só no treino.
#   3. Compara com os parâmetros antigos e com o baseline "sempre a média".
#   4. Mede o erro final no teste e salva as métricas em JSON.
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Parâmetros usados antes desta issue (referência para comparar)
PARAMS_ANTIGOS = dict(n_estimators=120, max_depth=10, min_samples_leaf=4)

espaco = {
    "n_estimators": [100, 200, 300],
    "max_depth": [6, 8, 10, 14, None],
    "min_samples_leaf": [2, 4, 8, 16],
    "max_features": [1.0, 0.8, 0.6],
}

busca = RandomizedSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=-1),
    param_distributions=espaco,
    n_iter=20,
    scoring="neg_mean_absolute_error",
    cv=cv,
    random_state=42,
    n_jobs=1,  # o paralelismo já está dentro do RandomForest
    refit=True,
)
busca.fit(X_train, y_train)

mae_cv_melhor = -busca.best_score_
modelo_antigo = RandomForestRegressor(random_state=42, n_jobs=-1, **PARAMS_ANTIGOS)
mae_cv_antigo = -cross_val_score(
    modelo_antigo, X_train, y_train, cv=cv, scoring="neg_mean_absolute_error"
).mean()

print("\nMelhores hiperparâmetros:", busca.best_params_)
print(f"MAE na validação cruzada: {mae_cv_melhor:.3f} dias "
      f"(parâmetros antigos: {mae_cv_antigo:.3f})")

# Avaliação FINAL, uma única vez, no teste que ficou de fora
pred = busca.best_estimator_.predict(X_test)
mae = mean_absolute_error(y_test, pred)
r2 = r2_score(y_test, pred)
baseline = mean_absolute_error(y_test, np.full_like(y_test, y_train.mean()))

# Faixa de incerteza (issue #13): percentis 10 e 90 do erro (real - previsto) no teste
erro = y_test - pred
setor_teste = X_test[:, FEATURES.index("setor")].astype(int)
intervalo_80 = {
    "geral": [round(float(np.percentile(erro, 10)), 3), round(float(np.percentile(erro, 90)), 3)],
    "por_setor": {
        nome: [
            round(float(np.percentile(erro[setor_teste == i], 10)), 3),
            round(float(np.percentile(erro[setor_teste == i], 90)), 3),
        ]
        for i, nome in enumerate(SETORES)
    },
}

print(f"MAE no teste: {mae:.2f} dias (baseline 'sempre a média': {baseline:.2f})")
print(f"R² no teste:  {r2:.3f}")

# Modelo final: mesmos hiperparâmetros, treinado com TODOS os dados
model = clone(busca.best_estimator_).fit(X, y)

out_dir = Path(__file__).parent / "models"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "modelo_alta.joblib"
joblib.dump(model, out_path)
print(f"Modelo salvo em: {out_path}")

import sklearn  # noqa: E402

metricas = {
    "gerado_em": datetime.now().isoformat(timespec="seconds"),
    "sklearn": sklearn.__version__,
    "amostras": int(len(y)),
    "melhores_hiperparametros": busca.best_params_,
    "mae_validacao_cruzada": round(float(mae_cv_melhor), 3),
    "mae_validacao_cruzada_params_antigos": round(float(mae_cv_antigo), 3),
    "mae_teste": round(float(mae), 3),
    "r2_teste": round(float(r2), 3),
    "mae_baseline": round(float(baseline), 3),
    "intervalo_80": intervalo_80,
}
(out_dir / "metricas_alta.json").write_text(
    json.dumps(metricas, indent=2, ensure_ascii=False), encoding="utf-8"
)
print(f"Métricas salvas em: {out_dir / 'metricas_alta.json'}")
