"""
Faixa de incerteza da previsão de alta (issue #13).

Em vez de um "nível de confiança" inventado, usamos o erro REAL do modelo no
conjunto de teste: o treino salva em metricas_alta.json os percentis 10 e 90
dos erros (real - previsto), geral e por setor. A faixa devolvida cobriu cerca
de 80% dos casos de teste.
"""
import json
from pathlib import Path

METRICAS_PATH = Path(__file__).parent / "models" / "metricas_alta.json"
PISO_DIAS = 0.5
COBERTURA = 0.8

_cache = {"mtime": None, "dados": None}


def _carregar():
    """Lê o bloco 'intervalo_80' das métricas; recarrega se o arquivo mudar."""
    try:
        mtime = METRICAS_PATH.stat().st_mtime
    except OSError:
        return None
    if _cache["mtime"] != mtime:
        try:
            dados = json.loads(METRICAS_PATH.read_text(encoding="utf-8")).get("intervalo_80")
        except (OSError, ValueError):
            dados = None
        _cache["mtime"], _cache["dados"] = mtime, dados
    return _cache["dados"]


def intervalo_previsao(dias, setor):
    """Devolve (minimo, maximo) em dias, ou None se não houver métricas salvas."""
    dados = _carregar()
    if not dados:
        return None
    nome = getattr(setor, "value", setor)
    limites = dados.get("por_setor", {}).get(nome) or dados.get("geral")
    if not limites:
        return None
    baixo, alto = limites
    return round(max(PISO_DIAS, dias + baixo), 1), round(max(PISO_DIAS, dias + alto), 1)