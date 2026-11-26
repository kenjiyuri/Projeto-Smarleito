"""
Cadastro de comorbidades (CID-10) do SmartLeito.

O catálogo fica em data/comorbidades.csv e já vem com as comorbidades mais comuns.
A equipe médica pode incluir novas informando:
    - CID-10 (ex.: I10 ou E11.9)
    - Nome da doença
    - Nome popular (opcional)

As funções deste módulo não dependem do Streamlit, o que facilita testá-las.
"""

from __future__ import annotations

import os
import re
import tempfile
import threading
import unicodedata
from pathlib import Path

import pandas as pd

CAMINHO_PADRAO = Path(__file__).resolve().parents[1] / "data" / "comorbidades.csv"

COLUNAS = ["cid", "nome", "nome_popular", "origem"]
ORIGEM_PADRAO = "Padrão"
ORIGEM_EQUIPE = "Cadastrada pela equipe"

TAM_MIN_NOME = 3
TAM_MAX_TEXTO = 120

# Uma letra, dois números e, opcionalmente, subcategoria (ex.: I10, E11.9, K35.80)
CID_RE = re.compile(r"^[A-Z]\d{2}(\.\d{1,2})?$")

# Textos que começam assim são interpretados como fórmula ao abrir o CSV no Excel
_INICIOS_PERIGOSOS = ("=", "+", "-", "@")

_trava = threading.Lock()

# Comorbidades pré-cadastradas: (CID-10, nome da doença, nome popular)
CATALOGO_PADRAO = [
    ("I10", "Hipertensão essencial (primária)", "Pressão alta"),
    ("E11", "Diabetes mellitus não insulino-dependente", "Diabetes tipo 2"),
    ("E10", "Diabetes mellitus insulino-dependente", "Diabetes tipo 1"),
    ("I25", "Doença isquêmica crônica do coração", "Doença das coronárias"),
    ("I50", "Insuficiência cardíaca", "Coração fraco"),
    ("I48", "Flutter e fibrilação atrial", "Arritmia"),
    ("I69", "Sequelas de doenças cerebrovasculares", "Sequela de derrame (AVC)"),
    ("J44", "Outras doenças pulmonares obstrutivas crônicas", "DPOC, enfisema"),
    ("J45", "Asma", "Asma, bronquite asmática"),
    ("N18", "Insuficiência renal crônica", "Rim fraco"),
    ("E66", "Obesidade", "Obesidade"),
    ("E78", "Distúrbios do metabolismo de lipoproteínas e outras lipidemias", "Colesterol alto"),
    ("E03", "Outro hipotireoidismo", "Tireoide preguiçosa"),
    ("F32", "Episódios depressivos", "Depressão"),
    ("F41", "Outros transtornos ansiosos", "Ansiedade"),
    ("F03", "Demência não especificada", "Demência, esquecimento grave"),
    ("G20", "Doença de Parkinson", "Mal de Parkinson"),
    ("G40", "Epilepsia", "Convulsões"),
    ("K74", "Fibrose e cirrose hepáticas", "Cirrose do fígado"),
    ("K21", "Doença de refluxo gastroesofágico", "Refluxo, azia"),
    ("D64", "Outras anemias", "Anemia"),
    ("D57", "Transtornos falciformes", "Anemia falciforme"),
    ("M81", "Osteoporose sem fratura patológica", "Ossos fracos"),
    ("C80", "Neoplasia maligna, sem especificação de localização", "Câncer"),
    ("Q21", "Malformações congênitas dos septos cardíacos", "Cardiopatia congênita"),
    ("Q90", "Síndrome de Down", "Síndrome de Down"),
]


# ---------------------------------------------------------------------------
# Normalização e validação
# ---------------------------------------------------------------------------
def normalizar_cid(valor) -> str:
    """Padroniza o CID: sem espaços, maiúsculo e com ponto (E119 vira E11.9)."""
    cid = re.sub(r"\s+", "", str(valor or "")).upper().replace(",", ".")
    if re.fullmatch(r"[A-Z]\d{3}", cid):
        cid = f"{cid[:3]}.{cid[3]}"
    return cid


def _limpar_texto(valor) -> str:
    return re.sub(r"\s+", " ", str(valor or "")).strip()


def _validar_texto(texto: str, campo: str, obrigatorio: bool) -> str:
    if not texto:
        if obrigatorio:
            raise ValueError(f"Informe o {campo}.")
        return ""
    if len(texto) > TAM_MAX_TEXTO:
        raise ValueError(f"O {campo} deve ter no máximo {TAM_MAX_TEXTO} caracteres.")
    if any(unicodedata.category(c).startswith("C") for c in texto):
        raise ValueError(f"O {campo} contém caracteres inválidos.")
    if texto.startswith(_INICIOS_PERIGOSOS):
        raise ValueError(f"O {campo} não pode começar com os símbolos = + - ou @.")
    if not any(c.isalpha() for c in texto):
        raise ValueError(f"O {campo} deve conter letras.")
    if obrigatorio and len(texto) < TAM_MIN_NOME:
        raise ValueError(f"O {campo} deve ter ao menos {TAM_MIN_NOME} caracteres.")
    return texto[0].upper() + texto[1:]


def validar(cid, nome, nome_popular="") -> tuple[str, str, str]:
    """Valida os três campos e devolve (cid, nome, nome_popular) padronizados.

    Levanta ValueError com uma mensagem clara quando algo estiver incorreto.
    """
    cid_ok = normalizar_cid(cid)
    if not cid_ok:
        raise ValueError("Informe o CID-10.")
    if not CID_RE.match(cid_ok):
        raise ValueError(
            "CID-10 inválido. Use uma letra seguida de dois números, com subcategoria "
            "opcional (exemplos: I10, E11.9)."
        )
    nome_ok = _validar_texto(_limpar_texto(nome), "nome da doença", obrigatorio=True)
    popular_ok = _validar_texto(_limpar_texto(nome_popular), "nome popular", obrigatorio=False)
    return cid_ok, nome_ok, popular_ok


# ---------------------------------------------------------------------------
# Leitura e gravação
# ---------------------------------------------------------------------------
def _gravar(df: pd.DataFrame, caminho: Path) -> None:
    """Grava de forma atômica: escreve em arquivo temporário e troca no final."""
    caminho.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=caminho.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8-sig", newline="") as f:
            df[COLUNAS].to_csv(f, index=False, lineterminator="\r\n")
        os.replace(tmp, caminho)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def _catalogo_inicial() -> pd.DataFrame:
    return pd.DataFrame(
        [(c, n, p, ORIGEM_PADRAO) for c, n, p in CATALOGO_PADRAO], columns=COLUNAS
    )


def carregar(caminho=None) -> pd.DataFrame:
    """Lê o catálogo (criando-o com as comorbidades padrão, se ainda não existir)."""
    caminho = Path(caminho) if caminho else CAMINHO_PADRAO
    with _trava:
        if not caminho.exists():
            _gravar(_catalogo_inicial(), caminho)
    return _ler(caminho)


def _ler(caminho: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(caminho, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    except (OSError, ValueError, pd.errors.ParserError):
        return _catalogo_inicial()
    for col in COLUNAS:
        if col not in df.columns:
            df[col] = ""
    df = df[COLUNAS].copy()
    df["cid"] = df["cid"].map(normalizar_cid)
    df = df[df["cid"] != ""].drop_duplicates(subset="cid", keep="first")
    return df.sort_values("nome", key=lambda s: s.map(_sem_acento)).reset_index(drop=True)


def adicionar(cid, nome, nome_popular="", caminho=None) -> dict:
    """Cadastra uma nova comorbidade. Levanta ValueError se for inválida ou repetida."""
    caminho = Path(caminho) if caminho else CAMINHO_PADRAO
    cid_ok, nome_ok, popular_ok = validar(cid, nome, nome_popular)
    with _trava:
        atual = _ler(caminho) if caminho.exists() else _catalogo_inicial()
        repetido = atual[atual["cid"] == cid_ok]
        if not repetido.empty:
            raise ValueError(
                f"O CID {cid_ok} já está cadastrado como “{repetido.iloc[0]['nome']}”."
            )
        nova = {
            "cid": cid_ok,
            "nome": nome_ok,
            "nome_popular": popular_ok,
            "origem": ORIGEM_EQUIPE,
        }
        _gravar(pd.concat([atual, pd.DataFrame([nova])], ignore_index=True), caminho)
    return nova


# ---------------------------------------------------------------------------
# Apoio à interface
# ---------------------------------------------------------------------------
def rotulo(cid: str, nome: str, nome_popular: str = "") -> str:
    """Texto exibido nas listas de seleção. Ex.: 'I10 — Hipertensão essencial · Pressão alta'."""
    texto = f"{cid} — {nome}"
    return f"{texto} · {nome_popular}" if nome_popular else texto


def _sem_acento(texto) -> str:
    base = unicodedata.normalize("NFKD", str(texto))
    return "".join(c for c in base if not unicodedata.combining(c)).casefold()


def buscar(df: pd.DataFrame, termo: str) -> pd.DataFrame:
    """Filtra por CID, nome da doença ou nome popular (ignora acentos e maiúsculas)."""
    termo = _sem_acento(_limpar_texto(termo))
    if not termo:
        return df
    alvo = (df["cid"] + " " + df["nome"] + " " + df["nome_popular"]).map(_sem_acento)
    return df[alvo.str.contains(termo, regex=False)]