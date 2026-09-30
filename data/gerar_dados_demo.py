"""Gera CSV de leitos e histórico de demanda para uso offline / testes."""

from pathlib import Path
from datetime import datetime, timedelta
import random
import csv

OUT = Path(__file__).parent
random.seed(42)

SETORES = ["UTI", "Enfermaria", "Emergência", "Cirúrgico", "Pediatria"]
STATUS = ["Livre", "Ocupado", "Em higienização", "Reservado", "Interditado"]
PESOS = [0.22, 0.48, 0.12, 0.10, 0.08]

# --- leitos.csv ---
with open(OUT / "leitos.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["id", "codigo", "setor", "status", "dias_internado", "previsao_alta"])
    for i in range(1, 49):
        st = random.choices(STATUS, weights=PESOS, k=1)[0]
        dias = random.randint(1, 18) if st == "Ocupado" else 0
        previsao = ""
        if st == "Ocupado":
            previsao = (datetime.now() + timedelta(days=random.randint(0, 5))).strftime("%Y-%m-%d")
        w.writerow([i, f"L-{i:03d}", random.choice(SETORES), st, dias, previsao])

print("Gerado: data/leitos.csv")

# --- demanda_historico.csv ---
with open(OUT / "demanda_historico.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["data", "demanda_prevista", "ocupacao_real"])
    for i in range(30):
        d = (datetime.now() - timedelta(days=29 - i)).strftime("%Y-%m-%d")
        dem = int(max(15, min(55, 35 + 8 * __import__("math").sin(i / 3) + random.gauss(0, 4))))
        ocu = int(max(10, min(48, dem * 0.92 + random.gauss(0, 2))))
        w.writerow([d, dem, ocu])

print("Gerado: data/demanda_historico.csv")
