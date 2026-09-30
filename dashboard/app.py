"""
SMARTLEITO — Dashboard acessível
Feito para ser fácil de usar por idosos e crianças:
letras grandes, botões grandes, cores fortes e linguagem simples.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random
import requests

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------
API_URL = "http://localhost:8000/api/v1"

st.set_page_config(
    page_title="SMARTLEITO — Fácil de usar",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS acessível: letras grandes, alto contraste, áreas de clique grandes
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Nunito', sans-serif !important;
        font-size: 20px !important;
    }

    .stApp {
        background: linear-gradient(160deg, #062a3a 0%, #0d5c7a 50%, #0a4a63 100%);
    }

    /* Sidebar grande e clara */
    section[data-testid="stSidebar"] {
        background: #062a3a !important;
        border-right: 3px solid #1a8aab;
        min-width: 280px !important;
    }
    section[data-testid="stSidebar"] * {
        color: #ffffff !important;
        font-size: 1.15rem !important;
    }
    section[data-testid="stSidebar"] .stRadio label {
        font-size: 1.25rem !important;
        padding: 0.6rem 0.4rem !important;
        line-height: 1.5 !important;
    }
    section[data-testid="stSidebar"] .stRadio label:hover {
        background: rgba(26, 138, 171, 0.35);
        border-radius: 10px;
    }

    /* Títulos bem grandes */
    h1 { font-size: 2.8rem !important; color: #ffffff !important; font-weight: 800 !important; }
    h2 { font-size: 2.1rem !important; color: #ffffff !important; font-weight: 700 !important; }
    h3 { font-size: 1.6rem !important; color: #ffffff !important; font-weight: 700 !important; }
    p, li, span, label { color: #f0f9ff !important; }

    /* Cards grandes e legíveis */
    .card {
        background: #0f4a63;
        border: 3px solid #1a8aab;
        border-radius: 20px;
        padding: 1.75rem 2rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
    }
    .card h3 { margin-top: 0; font-size: 1.5rem !important; }
    .card p { font-size: 1.2rem !important; line-height: 1.6 !important; color: #e0f2fe !important; }

    .hero-title {
        font-size: 3.8rem !important;
        font-weight: 800;
        text-align: center;
        color: #ffffff !important;
        margin: 1rem 0 0.5rem 0;
        letter-spacing: -0.02em;
    }
    .hero-sub {
        text-align: center;
        color: #b8e0f0 !important;
        font-size: 1.45rem !important;
        margin-bottom: 2rem;
        line-height: 1.5;
    }

    /* Caixas de número bem visíveis */
    .metric-box {
        background: #0f4a63;
        border: 3px solid #4ade80;
        border-radius: 18px;
        padding: 1.5rem 1rem;
        text-align: center;
    }
    .metric-box .value {
        font-size: 2.6rem !important;
        font-weight: 800;
        color: #ffffff !important;
    }
    .metric-box .label {
        font-size: 1.15rem !important;
        color: #b8e0f0 !important;
        margin-top: 0.4rem;
        line-height: 1.35;
    }

    /* Botões GRANDES */
    .stButton > button {
        background: #1a8aab !important;
        color: #ffffff !important;
        border: 3px solid #4ade80 !important;
        border-radius: 16px !important;
        font-weight: 700 !important;
        font-size: 1.3rem !important;
        padding: 0.9rem 2rem !important;
        min-height: 60px !important;
        width: 100%;
    }
    .stButton > button:hover {
        background: #4ade80 !important;
        color: #062a3a !important;
        border-color: #ffffff !important;
    }

    /* Inputs e selects maiores */
    .stSelectbox label, .stNumberInput label, .stSlider label, .stMultiSelect label {
        font-size: 1.2rem !important;
        font-weight: 700 !important;
        color: #ffffff !important;
    }
    div[data-baseweb="select"] > div,
    .stNumberInput input {
        font-size: 1.2rem !important;
        min-height: 52px !important;
    }

    /* Métricas Streamlit */
    div[data-testid="stMetric"] {
        background: #0f4a63;
        border: 3px solid #1a8aab;
        border-radius: 16px;
        padding: 1.25rem;
    }
    div[data-testid="stMetric"] label {
        color: #b8e0f0 !important;
        font-size: 1.1rem !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 2.2rem !important;
        font-weight: 800 !important;
    }

    /* Alertas legíveis */
    .stSuccess, .stInfo, .stWarning {
        font-size: 1.25rem !important;
        border-radius: 14px !important;
        padding: 1.2rem !important;
    }

    /* Tabela legível */
    .stDataFrame { font-size: 1.1rem !important; }

    /* Esconde menus desnecessários */
    #MainMenu, footer, header { visibility: hidden; }

    /* Botões de navegação rápida no topo */
    .nav-btn {
        display: inline-block;
        background: #1a8aab;
        color: #fff !important;
        border: 3px solid #4ade80;
        border-radius: 14px;
        padding: 0.85rem 1.4rem;
        margin: 0.35rem;
        font-size: 1.2rem;
        font-weight: 700;
        text-decoration: none;
        text-align: center;
        min-width: 140px;
    }

    .status-pill {
        display: inline-block;
        padding: 0.45rem 1rem;
        border-radius: 999px;
        font-weight: 700;
        font-size: 1.15rem;
        margin: 0.25rem 0;
    }
    .pill-livre { background: #166534; color: #bbf7d0; border: 2px solid #4ade80; }
    .pill-ocupado { background: #7f1d1d; color: #fecaca; border: 2px solid #f87171; }
    .pill-higiene { background: #713f12; color: #fde68a; border: 2px solid #fbbf24; }
    .pill-reservado { background: #1e3a5f; color: #bfdbfe; border: 2px solid #60a5fa; }
    .pill-interditado { background: #374151; color: #d1d5db; border: 2px solid #9ca3af; }

    .big-number {
        font-size: 3.5rem !important;
        font-weight: 800;
        color: #4ade80 !important;
        text-align: center;
        line-height: 1.1;
    }
    .big-label {
        font-size: 1.35rem !important;
        color: #e0f2fe !important;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Dados
# ---------------------------------------------------------------------------

def api_get(path: str):
    try:
        r = requests.get(f"{API_URL}{path}", timeout=2)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def api_post(path: str, payload: dict):
    try:
        r = requests.post(f"{API_URL}{path}", json=payload, timeout=3)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


@st.cache_data(ttl=30)
def carregar_leitos():
    data = api_get("/leitos")
    if data:
        return pd.DataFrame(data)
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


def gerar_historico():
    datas = pd.date_range(end=datetime.now(), periods=30, freq="D")
    demanda = np.clip(
        35 + np.sin(np.arange(30) / 3) * 8 + np.random.normal(0, 4, 30), 15, 55
    ).astype(int)
    ocupacao = np.clip(demanda * 0.92 + np.random.normal(0, 2, 30), 10, 48).astype(int)
    return pd.DataFrame({
        "Data": datas,
        "Demanda prevista": demanda,
        "Ocupação real": ocupacao,
    })


# ---------------------------------------------------------------------------
# Menu lateral — poucos itens, nomes simples e com emoji grande
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🏥 SMARTLEITO")
    st.markdown("### Menu")
    pagina = st.radio(
        "Escolha uma página",
        [
            "🏠 Início",
            "👀 Ver leitos",
            "📊 Números",
            "🔮 Quando o paciente vai embora?",
            "❓ O que é isso?",
            "👥 Quem fez",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        """
        <p style="font-size:1rem; color:#b8e0f0;">
        💡 Dica: use os botões grandes.<br>
        🔠 Letras grandes para ler fácil.<br>
        🟢 Verde = livre · 🔴 Vermelho = ocupado
        </p>
        """,
        unsafe_allow_html=True,
    )

# ===========================================================================
# PÁGINAS
# ===========================================================================

# ----- INÍCIO -----
if pagina == "🏠 Início":
    st.markdown('<p class="hero-title">🏥 SMARTLEITO</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-sub">Ajuda o hospital a cuidar dos leitos.<br>'
        "Fácil de ver. Fácil de entender.</p>",
        unsafe_allow_html=True,
    )

    st.markdown("### O que você pode fazer aqui?")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            """
            <div class="metric-box">
                <div class="value">👀</div>
                <div class="label">Ver quais leitos<br>estão livres</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="metric-box">
                <div class="value">📊</div>
                <div class="label">Ver os números<br>do hospital</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            """
            <div class="metric-box">
                <div class="value">🔮</div>
                <div class="label">Saber quando o<br>paciente pode ir embora</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="card">
            <h3>📌 Como usar</h3>
            <p>
            1. No menu à esquerda, toque na página que você quer.<br>
            2. Os botões são grandes — é só clicar.<br>
            3. <strong style="color:#4ade80;">Verde</strong> = leito livre ·
               <strong style="color:#f87171;">Vermelho</strong> = leito ocupado.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ----- VER LEITOS (mapa simples) -----
elif pagina == "👀 Ver leitos":
    st.markdown("## 👀 Ver leitos")
    st.markdown(
        '<p style="font-size:1.3rem; color:#b8e0f0;">Cada bolinha é um leito. Toque nos filtros se quiser.</p>',
        unsafe_allow_html=True,
    )

    df = carregar_leitos()
    status_col = "status" if "status" in df.columns else "Status"
    setor_col = "setor" if "setor" in df.columns else "Setor"
    cod_col = "codigo" if "codigo" in df.columns else "Leito"

    # Resumo em linguagem simples
    resumo = carregar_resumo()
    a, b, c = st.columns(3)
    with a:
        st.markdown(
            f'<div class="card" style="border-color:#4ade80; text-align:center;">'
            f'<p class="big-number">{resumo["livres"]}</p>'
            f'<p class="big-label">🟢 Livres agora</p></div>',
            unsafe_allow_html=True,
        )
    with b:
        st.markdown(
            f'<div class="card" style="border-color:#f87171; text-align:center;">'
            f'<p class="big-number" style="color:#f87171 !important;">{resumo["ocupados"]}</p>'
            f'<p class="big-label">🔴 Ocupados</p></div>',
            unsafe_allow_html=True,
        )
    with c:
        st.markdown(
            f'<div class="card" style="border-color:#fbbf24; text-align:center;">'
            f'<p class="big-number" style="color:#fbbf24 !important;">{resumo["em_higienizacao"]}</p>'
            f'<p class="big-label">🟡 Sendo limpos</p></div>',
            unsafe_allow_html=True,
        )

    st.markdown("### Filtros (opcional)")
    f1, f2 = st.columns(2)
    with f1:
        f_setor = st.multiselect(
            "Qual setor?",
            options=sorted(df[setor_col].unique()),
            default=sorted(df[setor_col].unique()),
        )
    with f2:
        f_status = st.multiselect(
            "Qual situação?",
            options=sorted(df[status_col].unique()),
            default=sorted(df[status_col].unique()),
        )

    filt = df[df[setor_col].isin(f_setor) & df[status_col].isin(f_status)].copy()

    # Grade visual de leitos (cards grandes)
    st.markdown("### Lista de leitos")
    cores_borda = {
        "Livre": "#4ade80",
        "Ocupado": "#f87171",
        "Em higienização": "#fbbf24",
        "Reservado": "#60a5fa",
        "Interditado": "#9ca3af",
    }
    emoji_status = {
        "Livre": "🟢",
        "Ocupado": "🔴",
        "Em higienização": "🟡",
        "Reservado": "🔵",
        "Interditado": "⚪",
    }

    # Mostrar em linhas de 4
    cols_per_row = 4
    rows_list = list(filt.itertuples())
    for i in range(0, len(rows_list), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            if i + j >= len(rows_list):
                break
            row = rows_list[i + j]
            status = getattr(row, status_col)
            codigo = getattr(row, cod_col)
            setor = getattr(row, setor_col)
            cor = cores_borda.get(status, "#1a8aab")
            em = emoji_status.get(status, "⬜")
            with col:
                st.markdown(
                    f"""
                    <div style="background:#0f4a63; border:4px solid {cor}; border-radius:16px;
                         padding:1.1rem; margin-bottom:0.8rem; text-align:center; min-height:120px;">
                        <div style="font-size:1.8rem; font-weight:800; color:#fff;">{codigo}</div>
                        <div style="font-size:1.15rem; color:#b8e0f0; margin:0.3rem 0;">{setor}</div>
                        <div style="font-size:1.25rem; font-weight:700;">{em} {status}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

# ----- NÚMEROS (dashboard simplificado) -----
elif pagina == "📊 Números":
    st.markdown("## 📊 Números do hospital")
    st.markdown(
        '<p style="font-size:1.3rem; color:#b8e0f0;">Veja de forma simples como estão os leitos hoje.</p>',
        unsafe_allow_html=True,
    )

    resumo = carregar_resumo()
    df = carregar_leitos()
    status_col = "status" if "status" in df.columns else "Status"
    setor_col = "setor" if "setor" in df.columns else "Setor"

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total de leitos", resumo["total_leitos"])
    k2.metric("🟢 Livres", resumo["livres"])
    k3.metric("🔴 Ocupados", resumo["ocupados"])
    k4.metric("% ocupado", f"{resumo['taxa_ocupacao']}%")

    st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Como estão os leitos?")
        sc = df[status_col].value_counts().reset_index()
        sc.columns = ["Situação", "Quantidade"]
        cores = {
            "Livre": "#4ade80",
            "Ocupado": "#f87171",
            "Em higienização": "#fbbf24",
            "Reservado": "#60a5fa",
            "Interditado": "#9ca3af",
        }
        fig = px.pie(
            sc,
            names="Situação",
            values="Quantidade",
            color="Situação",
            color_discrete_map=cores,
            hole=0.4,
        )
        fig.update_traces(textfont_size=18, textposition="inside")
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e0f2fe", size=16),
            legend=dict(font=dict(size=16, color="#e0f2fe")),
            margin=dict(t=20, b=20, l=20, r=20),
            height=420,
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Ocupados por setor")
        ocup = (
            df[df[status_col] == "Ocupado"]
            .groupby(setor_col)
            .size()
            .reset_index(name="Ocupados")
        )
        fig2 = px.bar(
            ocup,
            x=setor_col,
            y="Ocupados",
            color="Ocupados",
            color_continuous_scale=["#1a8aab", "#4ade80"],
            text="Ocupados",
        )
        fig2.update_traces(textfont_size=18, textposition="outside")
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e0f2fe", size=16),
            coloraxis_showscale=False,
            xaxis=dict(gridcolor="rgba(255,255,255,0.1)", title=""),
            yaxis=dict(gridcolor="rgba(255,255,255,0.1)", title=""),
            margin=dict(t=20, b=20, l=20, r=20),
            height=420,
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Últimos 30 dias")
    hist = gerar_historico()
    fig3 = go.Figure()
    fig3.add_trace(
        go.Scatter(
            x=hist["Data"],
            y=hist["Demanda prevista"],
            mode="lines+markers",
            name="Previsto",
            line=dict(color="#60a5fa", width=3),
            marker=dict(size=8),
        )
    )
    fig3.add_trace(
        go.Scatter(
            x=hist["Data"],
            y=hist["Ocupação real"],
            mode="lines+markers",
            name="Real",
            line=dict(color="#4ade80", width=3),
            marker=dict(size=8),
        )
    )
    fig3.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0f2fe", size=16),
        xaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.1)", title="Leitos"),
        legend=dict(orientation="h", y=1.12, font=dict(size=16)),
        margin=dict(t=50, b=20, l=20, r=20),
        height=400,
        hovermode="x unified",
    )
    st.plotly_chart(fig3, use_container_width=True)

# ----- PREVISÃO DE ALTA (formulário bem simples) -----
elif pagina == "🔮 Quando o paciente vai embora?":
    st.markdown("## 🔮 Quando o paciente pode ir embora?")
    st.markdown(
        '<p style="font-size:1.3rem; color:#b8e0f0;">'
        "Preencha os campos e aperte o botão grande. O computador calcula uma estimativa.</p>",
        unsafe_allow_html=True,
    )

    with st.form("form_alta"):
        st.markdown("### Dados do paciente")
        c1, c2 = st.columns(2)
        with c1:
            idade = st.number_input("Idade (anos)", min_value=0, max_value=120, value=55, step=1)
            dias_int = st.number_input(
                "Há quantos dias está internado?", min_value=0, max_value=60, value=3, step=1
            )
        with c2:
            setor = st.selectbox(
                "Em qual setor está?",
                ["UTI", "Enfermaria", "Emergência", "Cirúrgico", "Pediatria"],
            )
            complexidade = st.select_slider(
                "O quadro é leve ou grave?",
                options=[1, 2, 3, 4, 5],
                value=3,
                format_func=lambda x: {
                    1: "1 — Bem leve",
                    2: "2 — Leve",
                    3: "3 — Médio",
                    4: "4 — Grave",
                    5: "5 — Muito grave",
                }[x],
            )
        comorb = st.number_input(
            "Quantas outras doenças o paciente tem?", min_value=0, max_value=10, value=1, step=1
        )
        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("▶️  CALCULAR  —  quando pode ir embora?")

    if submitted:
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
            conf = result["confianca"]
        else:
            base = {
                "UTI": 7.5,
                "Enfermaria": 4.0,
                "Emergência": 1.5,
                "Cirúrgico": 5.0,
                "Pediatria": 3.5,
            }[setor]
            perm = base + max(0, (idade - 60) * 0.04) + (complexidade - 3) * 0.8 + comorb * 0.5
            dias = max(0.5, round(perm - dias_int, 1))
            data_p = (datetime.now() + timedelta(days=int(np.ceil(dias)))).strftime("%Y-%m-%d")
            conf = 0.65

        # Resultado bem destacado
        if dias <= 1:
            mensagem = "Pode ir embora em breve — talvez ainda hoje ou amanhã."
            cor_borda = "#4ade80"
        elif dias <= 3:
            mensagem = "Ainda precisa de alguns dias de cuidado."
            cor_borda = "#fbbf24"
        else:
            mensagem = "Ainda precisa de mais alguns dias no hospital."
            cor_borda = "#60a5fa"

        st.markdown(
            f"""
            <div class="card" style="border-color:{cor_borda}; text-align:center; margin-top:1.5rem;">
                <p style="font-size:1.4rem; color:#b8e0f0; margin-bottom:0.5rem;">Estimativa</p>
                <p class="big-number">{dias} dias</p>
                <p class="big-label">Data aproximada: <strong style="color:#fff;">{data_p}</strong></p>
                <p style="font-size:1.25rem; color:#e0f2fe; margin-top:1rem;">{mensagem}</p>
                <p style="font-size:1rem; color:#94c5d8; margin-top:0.8rem;">
                Isso é só uma ajuda. O médico decide de verdade.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ----- O QUE É ISSO (explicação simples) -----
elif pagina == "❓ O que é isso?":
    st.markdown("## ❓ O que é o SMARTLEITO?")
    st.markdown(
        """
        <div class="card">
            <h3>Em poucas palavras</h3>
            <p>
            O SMARTLEITO é um programa de computador que ajuda o hospital a saber:
            </p>
            <p>
            🟢 Quais camas (leitos) estão livres<br>
            🔴 Quais estão com paciente<br>
            🔮 Quando um paciente pode ir embora<br>
            📊 Se o hospital pode ficar cheio nos próximos dias
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="card">
            <h3>Por que isso importa?</h3>
            <p>
            Quando o hospital não planeja bem, as pessoas esperam muito na fila
            ou ficam sem leito. Com o SMARTLEITO, a equipe vê os números com clareza
            e se prepara antes de ficar lotado.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="card">
            <h3>Cores que usamos</h3>
            <p>
            <span class="status-pill pill-livre">🟢 Livre</span>
            &nbsp;pode receber paciente<br><br>
            <span class="status-pill pill-ocupado">🔴 Ocupado</span>
            &nbsp;já tem paciente<br><br>
            <span class="status-pill pill-higiene">🟡 Sendo limpo</span>
            &nbsp;aguardando limpeza<br><br>
            <span class="status-pill pill-reservado">🔵 Reservado</span>
            &nbsp;separado para alguém<br><br>
            <span class="status-pill pill-interditado">⚪ Interditado</span>
            &nbsp;não pode usar agora
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ----- EQUIPE -----
elif pagina == "👥 Quem fez":
    st.markdown('<p class="hero-title" style="font-size:3rem;">Obrigado!</p>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="card" style="text-align:center;">
            <p style="font-size:1.4rem; color:#e0f2fe; line-height:1.8;">
            <strong style="color:#fff;">João Pedro</strong> ·
            <strong style="color:#fff;">Kenji Yuri</strong> ·
            <strong style="color:#fff;">Luca Casari</strong><br>
            <strong style="color:#fff;">Lucia Maria</strong> ·
            <strong style="color:#fff;">Matheus Bargas</strong> ·
            <strong style="color:#fff;">Maria Luiza</strong>
            </p>
            <p style="font-size:1.2rem; color:#94c5d8; margin-top:1.2rem;">
            Unimar — Universidade de Marília
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="card" style="text-align:center;">
            <p style="font-size:1.25rem; color:#b8e0f0;">
            Este site foi feito para ser fácil de usar por todo mundo —
            inclusive idosos e crianças.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
