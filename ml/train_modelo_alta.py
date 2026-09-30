"""
Treina um Random Forest para prever dias restantes até a alta.

Gera dados sintéticos realistas e salva o modelo em ml/models/modelo_alta.joblib.

Uso:
    python ml/train_modelo_alta.py
"""

from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# ---------------------------------------------------------------------------
# Dados sintéticos
# Features: idade, dias_internado, complexidade (1-5), setor_encoded (0-4), comorbidades
# Target: dias_restantes até alta
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)
N = 5000

idade = RNG.integers(1, 95, N)
dias_internado = RNG.integers(0, 20, N)
complexidade = RNG.integers(1, 6, N)
setor = RNG.integers(0, 5, N)  # 0=UTI … 4=Pediatria
comorbidades = RNG.integers(0, 8, N)

# Permanência base por setor
base_setor = np.array([7.5, 4.0, 1.5, 5.0, 3.5])[setor]

permanencia_esperada = (
    base_setor
    + np.maximum(0, (idade - 60) * 0.04)
    + (complexidade - 3) * 0.8
    + comorbidades * 0.5
    + RNG.normal(0, 1.2, N)
)

dias_restantes = np.clip(permanencia_esperada - dias_internado, 0.5, 30)

X = np.column_stack([idade, dias_internado, complexidade, setor, comorbidades])
y = dias_restantes

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

print(f"MAE: {mae:.2f} dias")
print(f"R²:  {r2:.3f}")

out_dir = Path(__file__).parent / "models"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "modelo_alta.joblib"
joblib.dump(model, out_path)
print(f"Modelo salvo em: {out_path}")
