"""
SMARTLEITO — Sistema de Gestão Inteligente de Leitos
Interface institucional, inspirada na identidade visual do Hospital Beneficente Unimar.

As cores e os textos de identidade ficam concentrados na seção TEMA, logo abaixo,
para facilitar o ajuste ao padrão oficial da instituição.
"""

import os
import random
import sys
from datetime import date, datetime, timedelta
from html import escape
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
RAIZ_PROJETO = BASE_DIR.parent
# Só usa a API se a variável de ambiente estiver definida
# (ex.: SMARTLEITO_API=http://localhost:8000/api/v1)
API_URL = os.environ.get("SMARTLEITO_API", "")

for _p in (RAIZ_PROJETO, BASE_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import comorbidades as cmb  # noqa: E402
import setores as stor  # noqa: E402

from ml.incerteza import intervalo_previsao  # noqa: E402
from ml.preprocessing import SETORES, montar_features  # noqa: E402

MODEL_PATH = RAIZ_PROJETO / "ml" / "models" / "modelo_alta.joblib"
SETORES_ALTA = SETORES

# ---------------------------------------------------------------------------
# TEMA — ajuste aqui para acompanhar a identidade visual do hospital
# ---------------------------------------------------------------------------
TEMA = {
    "primaria": "#1B4067",        # azul-marinho (faixas, títulos)
    "primaria_escura": "#143874", # azul profundo
    "secundaria": "#309B96",      # verde-azulado (botões, destaques)
    "destaque": "#34A89D",        # verde-azulado claro (linhas e detalhes)
    "fundo": "#F4F8F8",
    "superficie": "#FFFFFF",
    "borda": "#DDE6E8",
    "texto": "#24323F",
    "texto_suave": "#5B6B7B",
}
NOME_SISTEMA = "SmartLeito"
SUBTITULO = "Gestão Inteligente de Leitos Hospitalares"
INSTITUICAO = "Hospital Beneficente Unimar"

COR_STATUS = {
    "Livre": "#2E7D32",
    "Ocupado": "#C62828",
    "Em higienização": "#B26A00",
    "Reservado": "#1565C0",
    "Interditado": "#616161",
}
FUNDO_STATUS = {
    "Livre": "#E8F5E9",
    "Ocupado": "#FDECEA",
    "Em higienização": "#FFF4E0",
    "Reservado": "#E3F0FD",
    "Interditado": "#EEEEEE",
}

PAGINAS = {
    "inicio": "Início",
    "leitos": "Leitos",
    "painel": "Painel",
    "setores": "Setores",
    "alta": "Previsão de Alta",
    "comorb": "Comorbidades",
    "sobre": "Institucional",
}


@st.cache_resource
def _carregar_modelo(path: str, mtime: float):
    try:
        return joblib.load(path)
    except Exception:
        return None


def carregar_modelo_alta():
    if not MODEL_PATH.exists():
        return None
    return _carregar_modelo(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)


st.set_page_config(
    page_title=f"{NOME_SISTEMA} | {INSTITUICAO}",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# CSS — layout institucional claro, tipografia legível
# ---------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@500;600;700;800&family=Open+Sans:wght@400;500;600;700&display=swap');

:root {
    --primaria: __primaria__;
    --primaria-escura: __primaria_escura__;
    --secundaria: __secundaria__;
    --destaque: __destaque__;
    --fundo: __fundo__;
    --superficie: __superficie__;
    --borda: __borda__;
    --texto: __texto__;
    --texto-suave: __texto_suave__;
    --sangria: calc(50% - 50vw);
    --sangria-pad: calc(50vw - 50%);
}

html, body, [class*="css"], .stApp { font-family: 'Open Sans', sans-serif !important; color: var(--texto); }
.stApp, [data-testid="stAppViewContainer"], section.main { background: var(--fundo); overflow-x: hidden; }

#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"],
[data-testid="stExpandSidebarButton"] { display: none !important; }
header[data-testid="stHeader"] { display: none !important; }
.block-container { padding-top: 0 !important; max-width: 1180px; padding-bottom: 0 !important; }

h1, h2, h3, h4 { font-family: 'Montserrat', sans-serif !important; color: var(--primaria) !important; }
h3 { font-size: 1.15rem !important; font-weight: 700 !important; }
p, li, label, span { color: var(--texto); }

/* ---------- Cabeçalho (barra branca, menu à direita) ---------- */
.st-key-topo {
    background: #fff; box-sizing: border-box; width: 100vw !important; max-width: 100vw !important;
    margin-left: calc(50% - 50vw) !important; margin-top: -1rem; padding: 0.4rem max(1rem, calc((100vw - 1020px) / 2));
    box-shadow: 0 2px 8px rgba(20, 56, 116, 0.10); position: relative; z-index: 5;
}
.marca { display: flex; align-items: center; gap: 0.7rem; padding: 0.5rem 0; }
.marca img {
    width: 150px;
    height: auto;
    display: block;
    object-fit: contain;
}
.marca-txt { line-height: 1; }
.marca-leg { font-family: 'Montserrat', sans-serif; font-size: 0.62rem; font-weight: 700; letter-spacing: .06em; color: var(--primaria); text-transform: uppercase; }
.marca-nome { font-family: 'Montserrat', sans-serif; font-weight: 800; font-size: 1.7rem; color: var(--secundaria); letter-spacing: -0.01em; margin-top: 3px; }

.st-key-topo div[data-testid="stRadio"] > div[role="radiogroup"] { justify-content: flex-end; gap: 0.2rem !important; flex-wrap: wrap; }
.st-key-topo div[data-testid="stRadio"] label {
    padding: 0.5rem 0.85rem !important; margin: 0 !important; border-radius: 999px; cursor: pointer; transition: background .15s;
}
.st-key-topo div[data-testid="stRadio"] label > div > div:first-child { display: none !important; }
.st-key-topo div[data-testid="stRadio"] label p {
    font-family: 'Montserrat', sans-serif; font-weight: 600; font-size: 0.92rem; color: var(--primaria) !important; margin: 0; white-space: nowrap;
}
.st-key-topo div[data-testid="stRadio"] label:hover { background: #E8F4F3; }
.st-key-topo div[data-testid="stRadio"] label:has(input:checked) { background: var(--secundaria); }
.st-key-topo div[data-testid="stRadio"] label:has(input:checked) p { color: #fff !important; font-weight: 700; }

/* ---------- Banner principal ---------- */
.hero {
    margin: 0 var(--sangria); padding: 3.2rem var(--sangria-pad) 3.6rem; position: relative; overflow: hidden;
    background: linear-gradient(105deg, var(--primaria-escura) 0%, var(--primaria) 60%, #24607F 100%);
}
.hero::before { /* triângulo decorativo */
    content: ""; position: absolute; right: 0; top: 0; width: 34%; height: 100%;
    background: linear-gradient(135deg, transparent 49.7%, rgba(255,255,255,.55) 49.8%, rgba(255,255,255,.55) 50.2%, transparent 50.3%),
                linear-gradient(135deg, rgba(48,155,150,.0) 50%, rgba(48,155,150,.35) 50.3%);
    pointer-events: none;
}
.hero-grid { position: relative; display: flex; justify-content: space-between; align-items: flex-end; gap: 2rem; flex-wrap: wrap; }
.hero-tag {
    display: inline-block; background: var(--destaque); color: #fff; font-family: 'Montserrat', sans-serif; font-weight: 800;
    font-size: 1.25rem; letter-spacing: .02em; text-transform: uppercase; padding: 0.7rem 4rem 0.7rem 1.6rem;
    clip-path: polygon(0 0, 100% 0, 92% 100%, 0 100%);
}
.hero-painel {
    background: #fff; padding: 1.6rem 4.5rem 1.8rem 1.6rem; clip-path: polygon(0 0, 100% 0, 94% 100%, 0 100%);
    max-width: 560px;
}
.hero-painel .l1, .hero-painel .l2, .hero-painel .l3 {
    font-family: 'Montserrat', sans-serif; font-weight: 800; line-height: 1.02; text-transform: uppercase; margin: 0; display: block;
}
.hero-painel .l1 { font-size: 2.6rem; color: var(--primaria-escura); }
.hero-painel .l2 { font-size: 3.1rem; color: var(--destaque); }
.hero-painel .l3 { font-size: 2.6rem; color: var(--primaria-escura); }
.hero-texto { color: #E6F2F2 !important; font-size: 1.02rem; line-height: 1.65; max-width: 540px; margin: 1.6rem 0 0 0; }
.hero-cta {
    display: block; background: var(--destaque); color: #fff !important; text-decoration: none !important;
    font-family: 'Montserrat', sans-serif; font-weight: 800; font-size: 1.6rem; line-height: 1.15; text-transform: uppercase;
    padding: 1.6rem 2.2rem 1.6rem 3.4rem; clip-path: polygon(14% 0, 100% 0, 100% 100%, 0 100%); transition: background .15s;
}
.hero-cta:hover { background: var(--secundaria); }

/* ---------- Faixa de atalhos ---------- */
.atalhos {
    display: grid; grid-template-columns: repeat(5, 1fr); background: var(--primaria);
    margin: 0 var(--sangria) 2.2rem; padding: 0 var(--sangria-pad); border-top: 4px solid var(--destaque);
}
.atalho {
    display: flex; align-items: center; gap: 0.8rem; padding: 2rem 1.1rem; text-decoration: none !important;
    border-left: 2px solid var(--secundaria); transition: background .15s;
}
.atalho:first-child { border-left: none; }
.atalho:hover { background: var(--primaria-escura); }
.atalho svg { width: 40px; height: 40px; flex: none; stroke: var(--secundaria); fill: none; stroke-width: 1.8; stroke-linecap: round; stroke-linejoin: round; }
.atalho span { color: #fff !important; font-family: 'Montserrat', sans-serif; font-weight: 600; font-size: 0.95rem; line-height: 1.3; }
@media (max-width: 900px) { .atalhos { grid-template-columns: repeat(2, 1fr); } .atalho { border-left: none; border-top: 2px solid var(--secundaria); } }
@media (max-width: 560px) { .atalhos { grid-template-columns: 1fr; } }

/* ---------- Títulos e cartões ---------- */
.titulo-pagina { margin: 2.2rem 0 0.2rem 0; }
.titulo-pagina h2 { margin: 0; font-size: 1.8rem !important; font-weight: 800 !important; }
.titulo-pagina h2::after { content: ""; display: block; width: 56px; height: 4px; background: var(--destaque); border-radius: 2px; margin-top: .55rem; }
.subtitulo-pagina { color: var(--texto-suave) !important; margin: 0.7rem 0 1.5rem 0; font-size: 1rem; }

.cartao {
    background: var(--superficie); border: 1px solid var(--borda); border-radius: 12px;
    padding: 1.4rem 1.6rem; margin-bottom: 1rem; box-shadow: 0 1px 3px rgba(16, 42, 67, 0.05);
}
.cartao h3 { margin-top: 0; }
.cartao p { line-height: 1.65; margin-bottom: 0.4rem; }
.cartao-destaque { border-left: 5px solid var(--destaque); }

.indicador {
    background: var(--superficie); border: 1px solid var(--borda); border-top: 4px solid var(--destaque);
    border-radius: 12px; padding: 1.2rem 1.3rem; box-shadow: 0 1px 3px rgba(16,42,67,.05);
}
.indicador .valor { font-family: 'Montserrat', sans-serif; font-weight: 800; font-size: 2.1rem; color: var(--primaria); line-height: 1.1; }
.indicador .rotulo { font-size: 0.8rem; color: var(--texto-suave); margin-top: 0.35rem; font-weight: 700; text-transform: uppercase; letter-spacing: .04em; font-family: 'Montserrat', sans-serif; }

div[data-testid="stMetric"] {
    background: var(--superficie); border: 1px solid var(--borda); border-top: 4px solid var(--destaque);
    border-radius: 12px; padding: 1rem 1.2rem; box-shadow: 0 1px 3px rgba(16,42,67,.05);
}
div[data-testid="stMetric"] label p { color: var(--texto-suave) !important; font-weight: 700; font-size: .8rem; text-transform: uppercase; letter-spacing: .04em; }
div[data-testid="stMetricValue"] { color: var(--primaria) !important; font-family: 'Montserrat', sans-serif; font-weight: 800; }

.selo { display: inline-block; padding: 0.25rem 0.75rem; border-radius: 999px; font-size: 0.82rem; font-weight: 700; margin: 0.15rem 0.2rem 0.15rem 0; border: 1px solid; }

.grade { display: grid; grid-template-columns: repeat(auto-fill, minmax(170px, 1fr)); gap: 0.8rem; }
.leito { background: var(--superficie); border: 1px solid var(--borda); border-left-width: 6px; border-radius: 10px; padding: 0.9rem 1rem; }
.leito .cod { font-family: 'Montserrat', sans-serif; font-weight: 800; color: var(--texto); font-size: 1.1rem; }
.leito .setor { color: var(--texto-suave); font-size: 0.88rem; margin: 0.1rem 0 0.5rem 0; }

div[data-testid="stForm"] { background: var(--superficie); border: 1px solid var(--borda); border-radius: 12px; padding: 1.5rem; }
.stSelectbox label, .stNumberInput label, .stSlider label, .stMultiSelect label, .stSelectSlider label { font-weight: 600 !important; color: var(--texto) !important; }

/* Botões em formato de pílula, como o "LOGIN" do site */
.stButton > button, .stFormSubmitButton > button {
    background: var(--secundaria) !important; color: #fff !important; border: none !important;
    border-radius: 999px !important; font-weight: 800 !important; padding: 0.7rem 2.2rem !important;
    font-family: 'Montserrat', sans-serif !important; letter-spacing: .06em; text-transform: uppercase;
}
.stButton > button:hover, .stFormSubmitButton > button:hover { background: var(--primaria) !important; }
.stFormSubmitButton > button p { color: #fff !important; }

.resultado { text-align: center; padding: 1.8rem 1.5rem; }
.resultado .rotulo { color: var(--texto-suave); font-weight: 700; text-transform: uppercase; letter-spacing: .05em; font-size: .82rem; font-family: 'Montserrat', sans-serif; }
.resultado .grande { font-family: 'Montserrat', sans-serif; font-weight: 800; font-size: 3rem; color: var(--primaria); line-height: 1.2; }
.aviso { font-size: 0.85rem; color: var(--texto-suave); margin-top: 1rem; }

div[data-testid="stMarkdownContainer"] h3 { margin-top: 0.8rem; }

/* ---------- Seletor de setor ---------- */
.st-key-seletor div[data-testid="stRadio"] > div[role="radiogroup"] { gap: 0.5rem !important; flex-wrap: wrap; }
.st-key-seletor div[data-testid="stRadio"] label {
    padding: 0.55rem 1.4rem !important; margin: 0 !important; border: 2px solid var(--secundaria);
    border-radius: 999px; background: #fff; cursor: pointer; transition: background .15s;
}
.st-key-seletor div[data-testid="stRadio"] label > div > div:first-child { display: none !important; }
.st-key-seletor div[data-testid="stRadio"] label p { color: var(--secundaria) !important; font-family: 'Montserrat', sans-serif; font-weight: 700; margin: 0; }
.st-key-seletor div[data-testid="stRadio"] label:hover { background: #E8F4F3; }
.st-key-seletor div[data-testid="stRadio"] label:has(input:checked) { background: var(--secundaria); }
.st-key-seletor div[data-testid="stRadio"] label:has(input:checked) p { color: #fff !important; }

/* ---------- Situação do setor, regras e tabelas ---------- */
.situacao { display: flex; align-items: center; gap: 1.2rem; border: 1px solid; border-left-width: 6px; border-radius: 12px; padding: 1rem 1.4rem; margin: 1.2rem 0; flex-wrap: wrap; }
.situacao .nivel { font-family: 'Montserrat', sans-serif; font-weight: 800; text-transform: uppercase; letter-spacing: .06em; font-size: .9rem; }
.situacao .msg { color: var(--texto); font-size: 0.98rem; }
.regras { display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 0.8rem; margin-bottom: 1rem; }
.regra { background: var(--superficie); border: 1px solid var(--borda); border-left: 5px solid; border-radius: 10px; padding: 0.9rem 1rem; }
.regra .t { font-family: 'Montserrat', sans-serif; font-weight: 700; color: var(--primaria); font-size: 0.95rem; }
.regra .s { font-size: .75rem; font-weight: 800; text-transform: uppercase; letter-spacing: .05em; margin: 0.25rem 0; }
.regra .d { color: var(--texto-suave); font-size: 0.9rem; line-height: 1.4; }
.tabela-rolagem { max-height: 520px; overflow: auto; border: 1px solid var(--borda); border-radius: 12px; background: var(--superficie); }
.tabela { width: 100%; border-collapse: collapse; font-size: 0.92rem; }
.tabela th { position: sticky; top: 0; background: var(--primaria); color: #fff; font-family: 'Montserrat', sans-serif; font-weight: 700; font-size: .8rem; text-transform: uppercase; letter-spacing: .04em; text-align: left; padding: .7rem .9rem; white-space: nowrap; }
.tabela td { padding: .65rem .9rem; border-top: 1px solid var(--borda); color: var(--texto); vertical-align: top; }
.tabela tr:nth-child(even) td { background: #F7FAFA; }
.tabela .cid { font-family: 'Montserrat', sans-serif; font-weight: 800; color: var(--secundaria); white-space: nowrap; }
.vazio { padding: 1.2rem; color: var(--texto-suave); text-align: center; }
.chips { margin-top: .6rem; }
.chip { display: inline-block; background: #E8F4F3; color: var(--primaria); border-radius: 999px; padding: .2rem .7rem; font-size: .85rem; font-weight: 600; margin: .15rem .25rem .15rem 0; }

/* ---------- Rodapé ---------- */
.rodape {
    margin: 3rem var(--sangria) 0; padding: 2rem var(--sangria-pad); background: var(--primaria);
    border-top: 4px solid var(--destaque); text-align: center; line-height: 1.7; font-size: .9rem;
}
.rodape, .rodape strong, .rodape span { color: #DCEBEC !important; }
.rodape strong { color: #fff !important; font-family: 'Montserrat', sans-serif; }
</style>
"""
for chave, valor in TEMA.items():
    CSS = CSS.replace(f"__{chave}__", valor)
st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Componentes de interface
# ---------------------------------------------------------------------------
def html(bloco: str):
    """Renderiza HTML sem indentação (evita que o Markdown trate como código)."""
    st.markdown("\n".join(l.strip() for l in bloco.strip().splitlines()), unsafe_allow_html=True)


def titulo_pagina(titulo: str, subtitulo: str = ""):
    html(
        f"""
        <div class="titulo-pagina"><h2>{titulo}</h2></div>
        <p class="subtitulo-pagina">{subtitulo}</p>
        """
    )


def indicador(valor, rotulo, cor=None):
    estilo = f' style="color:{cor};"' if cor else ""
    borda = f' style="border-top-color:{cor};"' if cor else ""
    return (
        f'<div class="indicador"{borda}><div class="valor"{estilo}>{valor}</div>'
        f'<div class="rotulo">{rotulo}</div></div>'
    )


def selo(status: str) -> str:
    cor = COR_STATUS.get(status, TEMA["primaria"])
    fundo = FUNDO_STATUS.get(status, "#EEE")
    return f'<span class="selo" style="color:{cor}; background:{fundo}; border-color:{cor};">{status}</span>'


class Bruto(str):
    """Texto já seguro (HTML pronto) para uso em tabela_html."""


def tabela_html(cabecalhos, linhas, vazio="Nenhum registro encontrado."):
    """Tabela institucional. Os textos são escapados, exceto os marcados como Bruto."""
    if not linhas:
        return f'<div class="tabela-rolagem"><div class="vazio">{escape(vazio)}</div></div>'
    cab = "".join(f"<th>{escape(c)}</th>" for c in cabecalhos)
    corpo = "".join(
        "<tr>" + "".join(f"<td>{c if isinstance(c, Bruto) else escape(str(c))}</td>" for c in linha) + "</tr>"
        for linha in linhas
    )
    return (f'<div class="tabela-rolagem"><table class="tabela"><thead><tr>{cab}</tr></thead>'
            f"<tbody>{corpo}</tbody></table></div>")


def estilo_grafico(fig, altura=400):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Open Sans, sans-serif", color=TEMA["texto"], size=14),
        margin=dict(t=30, b=30, l=20, r=20),
        height=altura,
    )
    fig.update_xaxes(gridcolor="#E6EBF0", linecolor=TEMA["borda"])
    fig.update_yaxes(gridcolor="#E6EBF0", linecolor=TEMA["borda"])
    return fig


def formulario_nova_comorbidade(chave: str, selecionar: bool = False, empilhado: bool = False):
    """Formulário para a equipe médica cadastrar uma nova comorbidade (CID, doença e nome popular).

    selecionar=True marca a nova comorbidade na lista do paciente. Nesse caso, chame-o
    ANTES do campo de seleção de comorbidades, na mesma página.
    empilhado=True coloca os campos um embaixo do outro (para colunas estreitas).
    """
    versao = st.session_state.get(f"_v_{chave}", 0)  # muda após cadastrar, para limpar os campos
    base = f"{chave}_{versao}"
    with st.form(base):
        c1, c2, c3 = (st, st, st) if empilhado else st.columns([1, 2, 2])
        cid = c1.text_input("CID-10", placeholder="Ex.: I10 ou E11.9", max_chars=8, key=f"{base}_cid")
        nome = c2.text_input("Nome da doença", max_chars=cmb.TAM_MAX_TEXTO, key=f"{base}_nome")
        popular = c3.text_input("Nome popular (opcional)", max_chars=cmb.TAM_MAX_TEXTO, key=f"{base}_pop")
        enviar = st.form_submit_button("Cadastrar comorbidade")
    if not enviar:
        return
    try:
        nova = cmb.adicionar(cid, nome, popular)
    except ValueError as erro:
        st.error(str(erro))
        return
    except OSError:
        st.error("Não foi possível gravar o cadastro. Verifique as permissões da pasta data/ e tente novamente.")
        return
    mensagem = f"Comorbidade {nova['cid']} cadastrada com sucesso."
    if selecionar:
        marcadas = list(st.session_state.get("comorb_sel", []))
        if len(marcadas) < 10:
            marcadas.append(cmb.rotulo(nova["cid"], nova["nome"], nova["nome_popular"]))
            st.session_state["comorb_sel"] = marcadas
            mensagem += " Ela já foi selecionada para este paciente."
    st.session_state["aviso_comorb"] = mensagem
    st.session_state[f"_v_{chave}"] = versao + 1
    st.rerun()


# ---------------------------------------------------------------------------
# Dados
# ---------------------------------------------------------------------------
def api_get(path: str):
    if not API_URL:
        return None
    try:
        r = requests.get(f"{API_URL}{path}", timeout=0.5)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def api_post(path: str, payload: dict):
    if not API_URL:
        return None
    try:
        r = requests.post(f"{API_URL}{path}", json=payload, timeout=3)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def _achar_csv(nome: str):
    """Procura o CSV na pasta do dashboard e na pasta data/ do projeto."""
    for pasta in (BASE_DIR, RAIZ_PROJETO / "data"):
        caminho = pasta / nome
        if caminho.exists():
            return caminho
    return None


@st.cache_data(ttl=30)
def carregar_leitos():
    data = api_get("/leitos")
    if data:
        return pd.DataFrame(data)
    csv_path = _achar_csv("leitos.csv")
    if csv_path:
        return pd.read_csv(csv_path)
    setores = ["UTI", "Enfermaria", "Emergência", "Cirúrgico", "Pediatria"]
    status_opts = ["Livre", "Ocupado", "Em higienização", "Reservado", "Interditado"]
    pesos = [0.22, 0.48, 0.12, 0.10, 0.08]
    rows = []
    for i in range(1, 49):
        st_ = random.choices(status_opts, weights=pesos, k=1)[0]
        dias = random.randint(1, 18) if st_ == "Ocupado" else 0
        prev = (
            (datetime.now() + timedelta(days=random.randint(0, 5))).strftime("%Y-%m-%d")
            if st_ == "Ocupado"
            else None
        )
        rows.append({
            "id": i,
            "codigo": f"L-{i:03d}",
            "setor": random.choice(setores),
            "status": st_,
            "dias_internado": dias,
            "previsao_alta": prev,
        })
    return pd.DataFrame(rows)


@st.cache_data(ttl=30)
def carregar_resumo():
    data = api_get("/leitos/resumo")
    if data:
        return data
    df = carregar_leitos()
    total = len(df)
    ocup = int((df["status"] == "Ocupado").sum())
    return {
        "total_leitos": total,
        "livres": int((df["status"] == "Livre").sum()),
        "ocupados": ocup,
        "em_higienizacao": int((df["status"] == "Em higienização").sum()),
        "reservados": int((df["status"] == "Reservado").sum()),
        "interditados": int((df["status"] == "Interditado").sum()),
        "taxa_ocupacao": round(ocup / total * 100, 1) if total else 0,
        "altas_previstas_5d": int(df["previsao_alta"].notna().sum()),
    }


@st.cache_data
def gerar_historico():
    csv_path = _achar_csv("demanda_historico.csv")
    if csv_path:
        h = pd.read_csv(csv_path, parse_dates=["data"])
        return h.rename(columns={
            "data": "Data",
            "demanda_prevista": "Demanda prevista",
            "ocupacao_real": "Ocupação real",
        })
    rng = np.random.default_rng(42)
    datas = pd.date_range(end=datetime.now(), periods=30, freq="D")
    demanda = np.clip(35 + np.sin(np.arange(30) / 3) * 8 + rng.normal(0, 4, 30), 15, 55).astype(int)
    ocupacao = np.clip(demanda * 0.92 + rng.normal(0, 2, 30), 10, 48).astype(int)
    return pd.DataFrame({"Data": datas, "Demanda prevista": demanda, "Ocupação real": ocupacao})


def colunas_leitos(df: pd.DataFrame):
    return (
        "status" if "status" in df.columns else "Status",
        "setor" if "setor" in df.columns else "Setor",
        "codigo" if "codigo" in df.columns else "Leito",
    )


# ---------------------------------------------------------------------------
# Cabeçalho e navegação
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Cabeçalho e navegação
# ---------------------------------------------------------------------------
LOGO_PATH = BASE_DIR / "logo_unimar.png"

if LOGO_PATH.exists():
    import base64

    with open(LOGO_PATH, "rb") as arquivo:
        logo_base64 = base64.b64encode(arquivo.read()).decode("utf-8")

    LOGO_HTML = (
        f'<img src="data:image/png;base64,{logo_base64}" '
        'alt="Logo Unimar">'
    )
else:
    LOGO_HTML = ""
# Permite que os atalhos da página inicial (links ?p=...) abram a página escolhida
if "pagina" not in st.session_state:
    st.session_state["pagina"] = "inicio"
_alvo = st.query_params.get("p")
if _alvo in PAGINAS:
    st.session_state["pagina"] = _alvo
    st.query_params.clear()

with st.container(key="topo"):
    col_marca, col_nav = st.columns([1, 3], vertical_alignment="center")

    with col_marca:
        html(
            f"""
            <div class="marca">
                {LOGO_HTML}
            </div>
            """
        )
    with col_nav:
        pagina = st.radio(
            "Navegação",
            list(PAGINAS.keys()),
            format_func=lambda k: PAGINAS[k],
            horizontal=True,
            label_visibility="collapsed",
            key="pagina",
        )

# ===========================================================================
# PÁGINAS
# ===========================================================================

# ----- INÍCIO -----
if pagina == "inicio":
    ICO_LEITO = '<svg viewBox="0 0 48 48"><path d="M5 38V12M5 28h38v10M5 28v-4a4 4 0 0 1 4-4h12a8 8 0 0 1 8 8M13 20a3 3 0 1 0 0-.1"/></svg>'
    ICO_GRAF = '<svg viewBox="0 0 48 48"><path d="M7 7v34h34M14 32l8-9 6 5 12-14M33 14h7v7"/></svg>'
    ICO_SETOR = '<svg viewBox="0 0 48 48"><path d="M6 42V10h20v32M26 42V20h16v22M4 42h40M12 18h4M12 26h4M12 34h4M32 28h4M32 36h4"/></svg>'
    ICO_ALTA = '<svg viewBox="0 0 48 48"><circle cx="24" cy="24" r="17"/><path d="M24 13v11l7 5"/></svg>'
    ICO_COMORB = '<svg viewBox="0 0 48 48"><path d="M16 8h16v6H16zM12 11H9v32h30V11h-3M24 22v12M18 28h12"/></svg>'
    html(
        f"""
        <div class="hero"><div class="hero-grid">
            <div>
                <span class="hero-tag">Gestão hospitalar</span>
                <div class="hero-painel">
                    <span class="l1">Gestão</span><span class="l2">Inteligente</span><span class="l3">de Leitos</span>
                </div>
                <p class="hero-texto">O {NOME_SISTEMA} apoia a equipe do {INSTITUICAO} no acompanhamento
                da ocupação, na previsão de altas e no planejamento da demanda de internações.</p>
            </div>
            <a class="hero-cta" href="?p=leitos" target="_self">Acesse o<br>mapa de leitos</a>
        </div></div>
        <div class="atalhos">
            <a class="atalho" href="?p=leitos" target="_self">{ICO_LEITO}<span>Mapa de<br>Leitos</span></a>
            <a class="atalho" href="?p=painel" target="_self">{ICO_GRAF}<span>Painel<br>Gerencial</span></a>
            <a class="atalho" href="?p=setores" target="_self">{ICO_SETOR}<span>Painel<br>por Setor</span></a>
            <a class="atalho" href="?p=alta" target="_self">{ICO_ALTA}<span>Previsão<br>de Alta</span></a>
            <a class="atalho" href="?p=comorb" target="_self">{ICO_COMORB}<span>Cadastro de<br>Comorbidades</span></a>
        </div>
        """
    )

    titulo_pagina("Situação atual dos leitos", "Resumo da ocupação hospitalar.")
    resumo = carregar_resumo()
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(indicador(resumo["total_leitos"], "Total de leitos"), unsafe_allow_html=True)
    k2.markdown(indicador(resumo["livres"], "Leitos livres", COR_STATUS["Livre"]), unsafe_allow_html=True)
    k3.markdown(indicador(resumo["ocupados"], "Leitos ocupados", COR_STATUS["Ocupado"]), unsafe_allow_html=True)
    k4.markdown(indicador(f"{stor.fmt(resumo['taxa_ocupacao'])}%", "Taxa de ocupação"), unsafe_allow_html=True)

    html(
        f"""
        <div class="cartao cartao-destaque">
            <h3>Legenda de situações</h3>
            <p>{selo("Livre")}{selo("Ocupado")}{selo("Em higienização")}{selo("Reservado")}{selo("Interditado")}</p>
        </div>
        """
    )

# ----- MAPA DE LEITOS -----
elif pagina == "leitos":
    titulo_pagina("Mapa de Leitos", "Situação atual dos leitos. Utilize os filtros para refinar a consulta.")

    df = carregar_leitos()
    status_col, setor_col, cod_col = colunas_leitos(df)
    resumo = carregar_resumo()

    a, b, c, d = st.columns(4)
    a.markdown(indicador(resumo["livres"], "Livres", COR_STATUS["Livre"]), unsafe_allow_html=True)
    b.markdown(indicador(resumo["ocupados"], "Ocupados", COR_STATUS["Ocupado"]), unsafe_allow_html=True)
    c.markdown(indicador(resumo["em_higienizacao"], "Em higienização", COR_STATUS["Em higienização"]), unsafe_allow_html=True)
    d.markdown(indicador(resumo["reservados"] + resumo["interditados"], "Indisponíveis", COR_STATUS["Reservado"]), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    f1, f2 = st.columns(2)
    with f1:
        f_setor = st.multiselect(
            "Setor",
            options=sorted(df[setor_col].unique()),
            default=sorted(df[setor_col].unique()),
        )
    with f2:
        f_status = st.multiselect(
            "Situação",
            options=sorted(df[status_col].unique()),
            default=sorted(df[status_col].unique()),
        )

    filt = df[df[setor_col].isin(f_setor) & df[status_col].isin(f_status)]
    st.markdown(f"**{len(filt)}** leito(s) exibido(s)")

    cards = []
    for row in filt.itertuples():
        status = getattr(row, status_col)
        cor = COR_STATUS.get(status, TEMA["primaria"])
        cards.append(
            f'<div class="leito" style="border-left-color:{cor};">'
            f'<div class="cod">{getattr(row, cod_col)}</div>'
            f'<div class="setor">{getattr(row, setor_col)}</div>{selo(status)}</div>'
        )
    st.markdown('<div class="grade">' + "".join(cards) + "</div>", unsafe_allow_html=True)

# ----- PAINEL GERENCIAL -----
elif pagina == "painel":
    titulo_pagina("Painel Gerencial", "Indicadores consolidados de ocupação dos leitos.")

    resumo = carregar_resumo()
    df = carregar_leitos()
    status_col, setor_col, _ = colunas_leitos(df)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total de leitos", resumo["total_leitos"])
    k2.metric("Leitos livres", resumo["livres"])
    k3.metric("Leitos ocupados", resumo["ocupados"])
    k4.metric("Taxa de ocupação", f"{stor.fmt(resumo['taxa_ocupacao'])}%")

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Distribuição por situação")
        sc = df[status_col].value_counts().reset_index()
        sc.columns = ["Situação", "Quantidade"]
        fig = px.pie(sc, names="Situação", values="Quantidade", color="Situação",
                     color_discrete_map=COR_STATUS, hole=0.5)
        fig.update_traces(textfont_size=14, textposition="inside")
        fig.update_layout(legend=dict(orientation="h", y=-0.1))
        st.plotly_chart(estilo_grafico(fig, 400), use_container_width=True)

    with c2:
        st.markdown("### Leitos ocupados por setor")
        ocup = (
            df[df[status_col] == "Ocupado"].groupby(setor_col).size().reset_index(name="Ocupados")
        )
        fig2 = px.bar(ocup, x=setor_col, y="Ocupados", text="Ocupados")
        fig2.update_traces(marker_color=TEMA["primaria"], textposition="outside")
        fig2.update_layout(xaxis_title="", yaxis_title="Leitos ocupados")
        st.plotly_chart(estilo_grafico(fig2, 400), use_container_width=True)

    st.markdown("### Ocupação por setor")
    base_setores = stor.normalizar_leitos(df)
    linhas_setor = []
    for nome_setor in stor.PERFIS:
        parte = base_setores[base_setores["setor"] == nome_setor]
        if len(parte):
            taxa = round((parte["status"] == "Ocupado").mean() * 100, 1)
            linhas_setor.append({"Setor": nome_setor, "Ocupação (%)": taxa,
                                 "Situação": stor.situacao(taxa, nome_setor)[0]})
    if linhas_setor:
        fig_s = px.bar(
            pd.DataFrame(linhas_setor), x="Ocupação (%)", y="Setor", orientation="h", color="Situação",
            text="Ocupação (%)",
            color_discrete_map={"Normal": COR_STATUS["Livre"], "Atenção": COR_STATUS["Em higienização"],
                                "Crítico": COR_STATUS["Ocupado"]},
        )
        fig_s.update_traces(textposition="outside")
        fig_s.update_layout(xaxis=dict(range=[0, 105], title="Ocupação (%)"), yaxis_title="",
                            legend=dict(orientation="h", y=1.15, title=""))
        st.plotly_chart(estilo_grafico(fig_s, 320), use_container_width=True)
        st.caption("A situação de cada setor segue os limites de referência definidos em dashboard/setores.py. "
                   "Para o detalhamento, acesse a página Setores.")

    st.markdown("### Demanda nos últimos 30 dias")
    hist = gerar_historico()
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=hist["Data"], y=hist["Demanda prevista"], mode="lines+markers", name="Demanda prevista",
        line=dict(color=TEMA["secundaria"], width=3, dash="dot"), marker=dict(size=6),
    ))
    fig3.add_trace(go.Scatter(
        x=hist["Data"], y=hist["Ocupação real"], mode="lines+markers", name="Ocupação real",
        line=dict(color=TEMA["primaria"], width=3), marker=dict(size=6),
    ))
    fig3.update_layout(
        yaxis_title="Leitos", legend=dict(orientation="h", y=1.12), hovermode="x unified",
    )
    st.plotly_chart(estilo_grafico(fig3, 400), use_container_width=True)

# ----- PAINEL POR SETOR -----
elif pagina == "setores":
    titulo_pagina(
        "Painel por Setor",
        "Indicadores, alertas e projeções específicos de cada área assistencial do hospital.",
    )

    leitos_df = stor.normalizar_leitos(carregar_leitos())
    historico = gerar_historico()
    ref = stor.data_referencia(leitos_df, historico["Data"].max().date())

    with st.container(key="seletor"):
        setor_sel = st.radio(
            "Setor", list(stor.PERFIS.keys()), horizontal=True,
            label_visibility="collapsed", key="setor_sel",
        )

    perfil = stor.PERFIS[setor_sel]
    dfs = leitos_df[leitos_df["setor"] == setor_sel]

    if dfs.empty:
        st.info(f"Não há leitos cadastrados no setor {setor_sel}.")
    else:
        m = stor.calcular(dfs, setor_sel, ref)
        nivel, chave_cor, mensagem_sit = stor.situacao(m["ocupacao"], setor_sel)
        cor_sit, fundo_sit = COR_STATUS[chave_cor], FUNDO_STATUS[chave_cor]
        ocupados_df = m["_ocupados_df"]

        html(
            f"""
            <div class="cartao cartao-destaque" style="margin-top:1.2rem;">
                <h3>{escape(setor_sel)}</h3>
                <p>{escape(perfil["descricao"])}</p>
                <p><strong>Foco do acompanhamento:</strong> {escape(perfil["foco"])}.</p>
            </div>
            <div class="situacao" style="background:{fundo_sit}; border-color:{cor_sit};">
                <span class="nivel" style="color:{cor_sit};">Situação: {nivel}</span>
                <span class="msg">{escape(mensagem_sit)}</span>
            </div>
            """
        )

        for col, (rotulo_kpi, valor_kpi) in zip(st.columns(4), stor.kpis(setor_sel, m)):
            col.markdown(indicador(valor_kpi, rotulo_kpi), unsafe_allow_html=True)

        st.markdown("### Pontos de acompanhamento")
        cartoes = []
        for r in stor.regras(setor_sel, m):
            cor_r = COR_STATUS["Livre"] if r["ok"] else COR_STATUS["Em higienização"]
            texto_r = "Adequado" if r["ok"] else "Requer atenção"
            cartoes.append(
                f'<div class="regra" style="border-left-color:{cor_r};">'
                f'<div class="t">{escape(r["titulo"])}</div>'
                f'<div class="s" style="color:{cor_r};">{texto_r}</div>'
                f'<div class="d">{escape(r["detalhe"])}</div></div>'
            )
        st.markdown('<div class="regras">' + "".join(cartoes) + "</div>", unsafe_allow_html=True)

        g1, g2 = st.columns(2)
        with g1:
            st.markdown("### Situação dos leitos")
            sc = dfs["status"].value_counts().reset_index()
            sc.columns = ["Situação", "Quantidade"]
            fig_a = px.pie(sc, names="Situação", values="Quantidade", color="Situação",
                           color_discrete_map=COR_STATUS, hole=0.5)
            fig_a.update_traces(textfont_size=14, textposition="inside")
            fig_a.update_layout(legend=dict(orientation="h", y=-0.1))
            st.plotly_chart(estilo_grafico(fig_a, 360), use_container_width=True, key=f"pizza_{setor_sel}")
        with g2:
            st.markdown("### Tempo de internação")
            if ocupados_df.empty:
                st.info("Não há pacientes internados neste setor.")
            else:
                maximo = int(ocupados_df["dias_internado"].max())
                fig_b = px.histogram(ocupados_df, x="dias_internado", nbins=min(10, max(3, maximo)))
                fig_b.update_traces(marker_color=TEMA["primaria"])
                fig_b.add_vline(
                    x=perfil["esperado"], line_dash="dash", line_color=COR_STATUS["Ocupado"],
                    annotation_text=f"Esperado: {stor.fmt(perfil['esperado'])} dias",
                    annotation_position="top right",
                )
                fig_b.update_layout(xaxis_title="Dias de internação", yaxis_title="Pacientes", bargap=0.1)
                fig_b.update_yaxes(dtick=1, rangemode="tozero")
                st.plotly_chart(estilo_grafico(fig_b, 360), use_container_width=True, key=f"hist_{setor_sel}")

        st.markdown("### Altas previstas e disponibilidade nos próximos 7 dias")
        proj = stor.projecao_altas(m, ref)
        fig_c = go.Figure()
        fig_c.add_trace(go.Bar(x=proj["data"], y=proj["altas"], name="Altas previstas",
                               marker_color=TEMA["secundaria"]))
        fig_c.add_trace(go.Scatter(x=proj["data"], y=proj["disponiveis"], mode="lines+markers",
                                   name="Leitos disponíveis (acumulado)",
                                   line=dict(color=TEMA["primaria"], width=3)))
        fig_c.update_layout(yaxis_title="Leitos", legend=dict(orientation="h", y=1.15), hovermode="x unified")
        st.plotly_chart(estilo_grafico(fig_c, 340), use_container_width=True, key=f"proj_{setor_sel}")
        st.caption("Leitos disponíveis = livres + em higienização + altas previstas acumuladas. "
                   "As altas em atraso são contadas no primeiro dia.")

        t1, t2 = st.columns(2)
        with t1:
            st.markdown(f"### Permanência acima de {stor.fmt(perfil['esperado'])} dias")
            acima = ocupados_df[ocupados_df["dias_internado"] > perfil["esperado"]].sort_values(
                "dias_internado", ascending=False)
            linhas_t1 = [
                [r.codigo, f"{int(r.dias_internado)} dias",
                 r.previsao_alta.strftime("%d/%m/%Y") if pd.notna(r.previsao_alta) else "—"]
                for r in acima.itertuples()
            ]
            st.markdown(tabela_html(["Leito", "Internado há", "Alta prevista"], linhas_t1,
                                    "Nenhum paciente acima da permanência esperada."), unsafe_allow_html=True)
        with t2:
            st.markdown("### Próximas altas")
            proximas = ocupados_df.dropna(subset=["previsao_alta"]).sort_values("previsao_alta").head(8)
            linhas_t2 = []
            for r in proximas.itertuples():
                if r.dias_ate_alta < 0:
                    quando = "Em atraso"
                elif r.dias_ate_alta == 0:
                    quando = "Hoje"
                else:
                    quando = f"Em {int(r.dias_ate_alta)} dia(s)"
                linhas_t2.append([r.codigo, r.previsao_alta.strftime("%d/%m/%Y"), quando])
            st.markdown(tabela_html(["Leito", "Data prevista", "Quando"], linhas_t2,
                                    "Nenhuma alta prevista."), unsafe_allow_html=True)

        st.markdown(f"### Leitos do setor ({len(dfs)})")
        cartoes_leito = []
        for r in dfs.itertuples():
            cor_l = COR_STATUS.get(r.status, TEMA["primaria"])
            cartoes_leito.append(
                f'<div class="leito" style="border-left-color:{cor_l};">'
                f'<div class="cod">{escape(str(r.codigo))}</div>'
                f'<div class="setor">{escape(setor_sel)}</div>{selo(r.status)}</div>'
            )
        st.markdown('<div class="grade">' + "".join(cartoes_leito) + "</div>", unsafe_allow_html=True)

        st.caption("Parâmetros de referência do setor (permanência esperada e limites de ocupação) definidos em "
                   "dashboard/setores.py. Ajuste-os conforme os protocolos do hospital.")
        if ref != date.today():
            st.caption(f"Os dados de demonstração foram gerados em {ref.strftime('%d/%m/%Y')}; "
                       "as previsões de alta são contadas a partir dessa data.")

# ----- PREVISÃO DE ALTA -----
elif pagina == "alta":
    titulo_pagina(
        "Previsão de Alta",
        "Informe os dados do paciente para obter uma estimativa do tempo restante de internação.",
    )

    catalogo = cmb.carregar()
    rotulos = [cmb.rotulo(r.cid, r.nome, r.nome_popular) for r in catalogo.itertuples()]
    dados_comorb = {rot: (r.cid, r.nome_popular) for rot, r in zip(rotulos, catalogo.itertuples())}
    # remove da seleção qualquer item que não exista mais no catálogo
    st.session_state["comorb_sel"] = [x for x in st.session_state.get("comorb_sel", []) if x in dados_comorb]

    aviso = st.session_state.pop("aviso_comorb", None)
    if aviso:
        st.success(aviso)
    with st.expander("A comorbidade do paciente não está na lista? Cadastre uma nova"):
        st.caption("Informe o CID-10, o nome da doença e, se desejar, o nome popular.")
        formulario_nova_comorbidade("nova_comorb_alta", selecionar=True)

    with st.form("form_alta"):
        st.markdown("### Dados do paciente")
        c1, c2 = st.columns(2)
        with c1:
            idade = st.number_input("Idade (anos)", min_value=0, max_value=120, value=55, step=1)
            dias_int = st.number_input(
                "Tempo de internação (dias)", min_value=0, max_value=365, value=3, step=1)
        with c2:
            setor = st.selectbox("Setor de internação", ["UTI", "Enfermaria", "Emergência", "Cirúrgico", "Pediatria"])
            complexidade = st.select_slider(
                "Gravidade do quadro clínico",
                options=[1, 2, 3, 4, 5],
                value=3,
                format_func=lambda x: {
                    1: "1 — Muito leve",
                    2: "2 — Leve",
                    3: "3 — Moderado",
                    4: "4 — Grave",
                    5: "5 — Muito grave",
                }[x],
            )
        comorb_sel = st.multiselect(
            "Comorbidades do paciente",
            options=rotulos,
            key="comorb_sel",
            max_selections=10,
            placeholder="Selecione uma ou mais comorbidades (opcional)",
            help="Busque pelo CID, pelo nome da doença ou pelo nome popular. É possível selecionar até 10.",
        )
        submitted = st.form_submit_button("Calcular previsão")

    if submitted:
        comorb = min(len(comorb_sel), 10)  # limite do modelo e da API
        payload = {
            "idade": int(idade),
            "dias_internado": int(dias_int),
            "complexidade": int(complexidade),
            "setor": setor,
            "comorbidades": int(comorb),
        }
        result = api_post("/previsao/alta", payload)

        if result:
            dias = result["dias_restantes"]
            data_p = result["data_prevista"]
            if result.get("intervalo_min") is not None and result.get("intervalo_max") is not None:
                faixa = (result["intervalo_min"], result["intervalo_max"])
            else:
                faixa = None
        else:
            modelo = carregar_modelo_alta()
            if modelo is not None:
                X = montar_features(int(idade), int(dias_int), int(complexidade), setor, int(comorb))
                dias = round(float(modelo.predict(X)[0]), 1)
                faixa = intervalo_previsao(dias, setor)
            else:
                base = {"UTI": 7.5, "Enfermaria": 4.0, "Emergência": 1.5, "Cirúrgico": 5.0, "Pediatria": 3.5}[setor]
                perm = base + max(0, (idade - 60) * 0.04) + (complexidade - 3) * 0.8 + comorb * 0.5
                # quem já passou do esperado ainda precisa de um tempo que cresce com a internação
                dias = round(max(perm - dias_int, 1 + 0.2 * dias_int), 1)
                faixa = None
            data_p = (datetime.now() + timedelta(days=int(np.ceil(dias)))).strftime("%d/%m/%Y")

        # Se a data vier da API em formato ISO, apresenta no padrão brasileiro
        try:
            data_p = datetime.strptime(str(data_p)[:10], "%Y-%m-%d").strftime("%d/%m/%Y")
        except ValueError:
            pass

        if dias <= 1:
            mensagem = "Alta prevista para as próximas 24 horas."
            cor = COR_STATUS["Livre"]
        elif dias <= 3:
            mensagem = "Paciente deve permanecer em acompanhamento por alguns dias."
            cor = COR_STATUS["Em higienização"]
        else:
            mensagem = "Paciente deve permanecer internado por período prolongado."
            cor = COR_STATUS["Reservado"]

        texto_faixa = ""
        if faixa:
            texto_faixa = (
                f"<p>Intervalo estimado: de <strong>{stor.fmt(faixa[0])}</strong> a "
                f"<strong>{stor.fmt(faixa[1])}</strong> dias.</p>"
            )

        if comorb:
            etiquetas = "".join(
                f'<span class="chip">{escape(dados_comorb[r][0])}'
                + (f" · {escape(dados_comorb[r][1])}" if dados_comorb[r][1] else "")
                + "</span>"
                for r in comorb_sel
            )
            texto_comorb = (
                f"<p><strong>Comorbidades informadas ({comorb}):</strong></p>"
                f'<div class="chips">{etiquetas}</div>'
                '<p class="aviso" style="margin-top:.4rem;">O modelo considera a quantidade de '
                "comorbidades informadas.</p>"
            )
        else:
            texto_comorb = "<p>Nenhuma comorbidade informada.</p>"

        html(
            f"""
            <div class="cartao resultado" style="border-top:5px solid {cor}; margin-top:1.2rem;">
                <div class="rotulo">Tempo restante estimado</div>
                <div class="grande">{stor.fmt(dias)} dias</div>
                <p>Data prevista para a alta: <strong>{data_p}</strong></p>
                <p>{mensagem}</p>
                {texto_faixa}
                {texto_comorb}
                <p class="aviso">Esta estimativa é apenas um instrumento de apoio ao planejamento.
                A decisão sobre a alta é sempre de responsabilidade da equipe médica.</p>
            </div>
            """
        )

# ----- COMORBIDADES -----
elif pagina == "comorb":
    titulo_pagina(
        "Cadastro de Comorbidades",
        "Consulte as comorbidades disponíveis no cadastro do paciente ou inclua uma nova, "
        "informando o CID-10, o nome da doença e o nome popular.",
    )

    aviso = st.session_state.pop("aviso_comorb", None)
    if aviso:
        st.success(aviso)

    catalogo = cmb.carregar()
    col_lista, col_form = st.columns([1.7, 1])

    with col_lista:
        termo = st.text_input(
            "Buscar comorbidade", placeholder="Digite o CID, o nome da doença ou o nome popular",
            max_chars=60,
        )
        filtrado = cmb.buscar(catalogo, termo)
        st.markdown(f"**{len(filtrado)}** de {len(catalogo)} comorbidade(s)")
        linhas_c = [
            [Bruto(f'<span class="cid">{escape(r.cid)}</span>'), r.nome, r.nome_popular or "—", r.origem]
            for r in filtrado.itertuples()
        ]
        st.markdown(
            tabela_html(["CID-10", "Doença", "Nome popular", "Origem"], linhas_c,
                        "Nenhuma comorbidade encontrada para essa busca."),
            unsafe_allow_html=True,
        )

    with col_form:
        html(
            """
            <div class="cartao cartao-destaque" style="margin-top:1.6rem;">
                <h3>Nova comorbidade</h3>
                <p>Não encontrou a doença na lista? Cadastre-a aqui e ela ficará disponível
                para todos os pacientes.</p>
                <p class="aviso">Confira o código na tabela oficial da CID-10. Cada CID só pode ser cadastrado uma vez.</p>
            </div>
            """
        )
        formulario_nova_comorbidade("nova_comorb_pagina", empilhado=True)

# ----- INSTITUCIONAL -----
elif pagina == "sobre":
    titulo_pagina("Institucional", f"Conheça o {NOME_SISTEMA} e a equipe responsável pelo projeto.")

    html(
        f"""
        <div class="cartao cartao-destaque">
            <h3>Sobre o sistema</h3>
            <p>O {NOME_SISTEMA} é uma ferramenta de apoio à gestão hospitalar que reúne, em um único
            painel, a situação dos leitos, a previsão de altas e a projeção da demanda por internações.</p>
            <p>Com informações organizadas e atualizadas, a equipe se antecipa a períodos de maior
            lotação e reduz o tempo de espera dos pacientes.</p>
        </div>
        """
    )
    html(
        f"""
        <div class="cartao">
            <h3>Situações dos leitos</h3>
            <p>{selo("Livre")}{selo("Ocupado")}{selo("Em higienização")}{selo("Reservado")}{selo("Interditado")}</p>
        </div>
        """
    )
    html(
        """
        <div class="cartao" style="text-align:center;">
            <h3>Equipe do projeto</h3>
            <p><strong>João Pedro</strong> · <strong>Kenji Yuri</strong> · <strong>Luca Casari</strong><br>
            <strong>Lucia Maria</strong> · <strong>Matheus Bargas</strong> · <strong>Maria Luiza</strong></p>
            <p style="color:#5B6B7B;">Unimar — Universidade de Marília</p>
        </div>
        """
    )

# ---------------------------------------------------------------------------
# Rodapé
# ---------------------------------------------------------------------------
html(
    f"""
    <div class="rodape">
        <strong>{NOME_SISTEMA}</strong> · {SUBTITULO}<br>
        Projeto acadêmico da Universidade de Marília (Unimar) · Marília, São Paulo<br>
        <span>Protótipo educacional com dados sintéticos. Não deve ser utilizado para decisões clínicas reais.</span>
    </div>
    """
)