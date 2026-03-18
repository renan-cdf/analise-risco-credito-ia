# =============================================================================
# PROJETO: Dashboard de Análise de Crédito com IA
# Interface: Streamlit
# Autor: Projeto Portfólio - Analista de Dados com IA | Sicoob
# =============================================================================

# Instalação:
# pip install streamlit plotly groq python-dotenv

# Para rodar:
# streamlit run app.py

import os
import joblib
import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from groq import Groq
from dotenv import load_dotenv
warnings.filterwarnings("ignore")

load_dotenv()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -----------------------------------------------------------------------------
# SETUP AUTOMÁTICO — gera dados e modelo se não existirem
# -----------------------------------------------------------------------------
def setup_automatico():
    """Coleta dados do BACEN e treina o modelo se os arquivos não existirem."""
    from bcb import sgs
    from datetime import datetime
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestRegressor

    arquivos = ["indicadores_bacen_bruto.csv", "features_bacen.csv",
                "modelo_inadimplencia.pkl", "scaler.pkl", "features.pkl"]
    caminhos = [os.path.join(BASE_DIR, f) for f in arquivos]

    if all(os.path.exists(c) for c in caminhos):
        return  # Tudo já existe, nada a fazer

    st.info("⚙️ Primeira execução — coletando dados do BACEN e treinando modelo...")
    progress = st.progress(0, text="Conectando ao BACEN...")

    # 1. Coleta de dados
    DATA_INICIO = "2015-03-18"
    DATA_FIM    = datetime.today().strftime("%Y-%m-%d")

    indicadores = sgs.get(
        {
            "IPCA (%)":            433,
            "Inadimplência (%)":   21082,
            "Juros PF (% a.a.)":   20714,
            "Selic Meta (% a.a.)": 4189,
        },
        start=DATA_INICIO, end=DATA_FIM,
    )
    indicadores_clean = indicadores.ffill().bfill()
    indicadores_clean.to_csv(os.path.join(BASE_DIR, "indicadores_bacen_bruto.csv"))
    progress.progress(30, text="Dados coletados! Criando features...")

    # 2. Feature engineering
    df = indicadores_clean.copy()
    for col in indicadores_clean.columns:
        df[f'{col}_var_1m'] = df[col].pct_change(1) * 100
        df[f'{col}_var_3m'] = df[col].pct_change(3) * 100
        df[f'{col}_mm3']    = df[col].rolling(3).mean()
        df[f'{col}_mm6']    = df[col].rolling(6).mean()
        df[f'{col}_mm12']   = df[col].rolling(12).mean()
        df[f'{col}_lag1']   = df[col].shift(1)
        df[f'{col}_lag3']   = df[col].shift(3)
    df = df.dropna()
    df.to_csv(os.path.join(BASE_DIR, "features_bacen.csv"))
    progress.progress(60, text="Features criadas! Treinando modelo...")

    # 3. Treinamento
    TARGET   = "Inadimplência (%)"
    cols_drop = [c for c in df.columns if "Inadimplência" in c and c != TARGET]
    df_model  = df.drop(columns=cols_drop)
    FEATURES  = [c for c in df_model.columns if c != TARGET]
    X = df_model[FEATURES]
    y = df_model[TARGET]

    split = int(len(df_model) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train         = y.iloc[:split]

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)

    modelo = RandomForestRegressor(n_estimators=200, random_state=42)
    modelo.fit(X_train_sc, y_train)

    joblib.dump(modelo,   os.path.join(BASE_DIR, "modelo_inadimplencia.pkl"))
    joblib.dump(scaler,   os.path.join(BASE_DIR, "scaler.pkl"))
    joblib.dump(FEATURES, os.path.join(BASE_DIR, "features.pkl"))
    progress.progress(100, text="✅ Pronto!")
    st.success("Dados e modelo carregados com sucesso!")
    st.rerun()

# -----------------------------------------------------------------------------
# CONFIGURAÇÃO DA PÁGINA
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Análise de Crédito | Sicoob",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS customizado
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background-color: #0f1117;
        color: #e8eaf0;
    }

    .metric-card {
        background: linear-gradient(135deg, #1a1d2e 0%, #16192a 100%);
        border: 1px solid #2a2d3e;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 600;
        color: #4fc3f7;
        font-family: 'DM Mono', monospace;
    }

    .metric-label {
        font-size: 0.8rem;
        color: #7b8099;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-top: 4px;
    }

    .metric-delta-up {
        font-size: 0.85rem;
        color: #ef5350;
        margin-top: 6px;
    }

    .metric-delta-down {
        font-size: 0.85rem;
        color: #66bb6a;
        margin-top: 6px;
    }

    .chat-user {
        background: #1e2235;
        border-left: 3px solid #4fc3f7;
        border-radius: 0 8px 8px 0;
        padding: 12px 16px;
        margin: 8px 0;
        font-size: 0.95rem;
    }

    .chat-bot {
        background: #161926;
        border-left: 3px solid #66bb6a;
        border-radius: 0 8px 8px 0;
        padding: 12px 16px;
        margin: 8px 0;
        font-size: 0.95rem;
        line-height: 1.6;
    }

    .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #c5cae9;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 16px;
        padding-bottom: 8px;
        border-bottom: 1px solid #2a2d3e;
    }

    .tag {
        display: inline-block;
        background: #1a2744;
        color: #4fc3f7;
        border: 1px solid #1e3a6e;
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 0.78rem;
        margin: 3px;
        cursor: pointer;
    }

    div[data-testid="stSidebar"] {
        background-color: #0d0f1a;
        border-right: 1px solid #1e2235;
    }

    .stTextInput input {
        background-color: #1a1d2e !important;
        border: 1px solid #2a2d3e !important;
        color: #e8eaf0 !important;
        border-radius: 8px !important;
        font-family: 'DM Sans', sans-serif !important;
    }

    .stButton button {
        background: linear-gradient(135deg, #1565c0, #0d47a1);
        color: white;
        border: none;
        border-radius: 8px;
        font-family: 'DM Sans', sans-serif;
        font-weight: 500;
    }

    .stButton button:hover {
        background: linear-gradient(135deg, #1976d2, #1565c0);
    }

    h1, h2, h3 { color: #e8eaf0 !important; }

    .previsao-box {
        background: linear-gradient(135deg, #0d2137 0%, #0a1a2e 100%);
        border: 1px solid #1565c0;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# CARREGAR DADOS E MODELO
# -----------------------------------------------------------------------------
@st.cache_data
def carregar_dados():
    df_bruto = pd.read_csv(
        os.path.join(BASE_DIR, "indicadores_bacen_bruto.csv"),
        index_col=0, parse_dates=True
    )
    df_features = pd.read_csv(
        os.path.join(BASE_DIR, "features_bacen.csv"),
        index_col=0, parse_dates=True
    )
    return df_bruto, df_features

@st.cache_resource
def carregar_modelo():
    modelo   = joblib.load(os.path.join(BASE_DIR, "modelo_inadimplencia.pkl"))
    scaler   = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
    features = joblib.load(os.path.join(BASE_DIR, "features.pkl"))
    return modelo, scaler, features

setup_automatico()
df_bruto, df_features = carregar_dados()
modelo, scaler, features = carregar_modelo()

# -----------------------------------------------------------------------------
# FUNÇÕES AUXILIARES
# -----------------------------------------------------------------------------
def fazer_previsao():
    ultima = df_features[features].iloc[[-1]]
    X_sc   = scaler.transform(ultima)
    return modelo.predict(X_sc)[0]

def gerar_contexto():
    stats    = df_bruto.describe().round(3).to_string()
    ultimos  = df_bruto.tail(6).round(3).to_string()
    corr     = df_bruto.corr().round(3).to_string()
    prev     = fazer_previsao()

    return f"""
Você é um analista de dados especialista em mercado financeiro cooperativista,
com foco em risco de crédito e indicadores macroeconômicos do Brasil.
Responda sempre em português, de forma clara, objetiva e profissional.

## Dados disponíveis (BACEN):
Período: {df_bruto.index[0].strftime('%b/%Y')} → {df_bruto.index[-1].strftime('%b/%Y')} ({len(df_bruto)} meses)

Indicadores: IPCA (%), Inadimplência (%), Juros PF (% a.a.), Selic Meta (% a.a.)

## Estatísticas:
{stats}

## Últimos 6 meses:
{ultimos}

## Correlações:
{corr}

## Modelo preditivo (Random Forest):
- R²: 0.556 | MAE: 0.219 pp
- Previsão próximo mês: {prev:.2f}%
- Último valor real: {df_bruto['Inadimplência (%)'].iloc[-1]:.2f}%

Se a pergunta não for sobre esses dados, informe educadamente.
"""

def chamar_groq(historico):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "❌ Chave GROQ_API_KEY não encontrada no arquivo .env"
    client = Groq(api_key=api_key)
    resposta = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": gerar_contexto()},
            *historico,
        ],
        max_tokens=1024,
        temperature=0.3,
    )
    return resposta.choices[0].message.content

# -----------------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🏦 Análise de Crédito")
    st.markdown("<small style='color:#7b8099'>Indicadores BACEN + IA</small>", unsafe_allow_html=True)
    st.divider()

    pagina = st.radio(
        "Navegação",
        ["📊 Dashboard", "🤖 Agente IA", "🔮 Previsão"],
        label_visibility="collapsed"
    )

    st.divider()

    ultimo = df_bruto.iloc[-1]
    anterior = df_bruto.iloc[-2]

    st.markdown("<div class='section-title'>Último registro</div>", unsafe_allow_html=True)
    for col in df_bruto.columns:
        delta = ultimo[col] - anterior[col]
        sinal = "▲" if delta > 0 else "▼"
        cor   = "#ef5350" if delta > 0 else "#66bb6a"
        st.markdown(
            f"<div style='display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid #1e2235'>"
            f"<span style='color:#9099b5;font-size:0.8rem'>{col.split('(')[0].strip()}</span>"
            f"<span style='color:#e8eaf0;font-family:DM Mono,monospace;font-size:0.85rem'>"
            f"{ultimo[col]:.2f} <span style='color:{cor}'>{sinal}{abs(delta):.2f}</span></span>"
            f"</div>",
            unsafe_allow_html=True
        )

    st.divider()
    st.markdown(f"<small style='color:#7b8099'>Dados: {df_bruto.index[0].strftime('%b/%Y')} → {df_bruto.index[-1].strftime('%b/%Y')}</small>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# PÁGINA: DASHBOARD
# -----------------------------------------------------------------------------
if "📊 Dashboard" in pagina:
    st.markdown("## 📊 Indicadores Econômicos — BACEN")

    # Métricas no topo
    cols = st.columns(4)
    nomes = ["IPCA (%)", "Inadimplência (%)", "Juros PF (% a.a.)", "Selic Meta (% a.a.)"]
    icones = ["📈", "⚠️", "💳", "🏛️"]

    for col, nome, icone in zip(cols, nomes, icones):
        val   = df_bruto[nome].iloc[-1]
        delta = df_bruto[nome].iloc[-1] - df_bruto[nome].iloc[-2]
        sinal = "▲" if delta > 0 else "▼"
        cor   = "metric-delta-up" if delta > 0 else "metric-delta-down"
        col.markdown(
            f"<div class='metric-card'>"
            f"<div style='font-size:1.5rem'>{icone}</div>"
            f"<div class='metric-value'>{val:.2f}%</div>"
            f"<div class='metric-label'>{nome}</div>"
            f"<div class='{cor}'>{sinal} {abs(delta):.2f} pp</div>"
            f"</div>",
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Gráfico principal — Inadimplência + Selic
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("<div class='section-title'>Inadimplência vs Selic</div>", unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_bruto.index, y=df_bruto["Inadimplência (%)"],
            name="Inadimplência", line=dict(color="#ef5350", width=2),
            fill='tozeroy', fillcolor='rgba(239,83,80,0.08)'
        ))
        fig.add_trace(go.Scatter(
            x=df_bruto.index, y=df_bruto["Selic Meta (% a.a.)"],
            name="Selic Meta", line=dict(color="#4fc3f7", width=2, dash="dash"),
            yaxis="y2"
        ))
        fig.update_layout(
            paper_bgcolor="#1a1d2e", plot_bgcolor="#1a1d2e",
            font=dict(color="#9099b5", family="DM Sans"),
            yaxis=dict(title="Inadimplência (%)", gridcolor="#2a2d3e", color="#ef5350"),
            yaxis2=dict(title="Selic (%)", overlaying="y", side="right", color="#4fc3f7", gridcolor="#2a2d3e"),
            legend=dict(bgcolor="#1a1d2e", bordercolor="#2a2d3e"),
            margin=dict(l=10, r=10, t=10, b=10),
            height=320,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("<div class='section-title'>Correlação</div>", unsafe_allow_html=True)
        corr = df_bruto.corr().round(2)
        labels_curtos = ["IPCA", "Inadimpl.", "Juros PF", "Selic"]
        fig_corr = go.Figure(go.Heatmap(
            z=corr.values, x=labels_curtos, y=labels_curtos,
            colorscale="RdBu_r", zmid=0,
            text=corr.values, texttemplate="%{text}",
        ))
        fig_corr.update_layout(
            paper_bgcolor="#1a1d2e", plot_bgcolor="#1a1d2e",
            font=dict(color="#9099b5", family="DM Sans"),
            margin=dict(l=10, r=10, t=10, b=10),
            height=320,
        )
        st.plotly_chart(fig_corr, use_container_width=True)

    # Gráfico IPCA e Juros
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("<div class='section-title'>IPCA — Inflação Mensal</div>", unsafe_allow_html=True)
        fig2 = go.Figure(go.Bar(
            x=df_bruto.index[-36:], y=df_bruto["IPCA (%)"].iloc[-36:],
            marker_color=["#ef5350" if v > 0 else "#66bb6a" for v in df_bruto["IPCA (%)"].iloc[-36:]]
        ))
        fig2.update_layout(
            paper_bgcolor="#1a1d2e", plot_bgcolor="#1a1d2e",
            font=dict(color="#9099b5"), yaxis=dict(gridcolor="#2a2d3e"),
            margin=dict(l=10, r=10, t=10, b=10), height=260
        )
        st.plotly_chart(fig2, use_container_width=True)

    with col4:
        st.markdown("<div class='section-title'>Juros PF (% a.a.)</div>", unsafe_allow_html=True)
        fig3 = go.Figure(go.Scatter(
            x=df_bruto.index, y=df_bruto["Juros PF (% a.a.)"],
            line=dict(color="#ffa726", width=2),
            fill='tozeroy', fillcolor='rgba(255,167,38,0.08)'
        ))
        fig3.update_layout(
            paper_bgcolor="#1a1d2e", plot_bgcolor="#1a1d2e",
            font=dict(color="#9099b5"), yaxis=dict(gridcolor="#2a2d3e"),
            margin=dict(l=10, r=10, t=10, b=10), height=260
        )
        st.plotly_chart(fig3, use_container_width=True)

# -----------------------------------------------------------------------------
# PÁGINA: AGENTE IA
# -----------------------------------------------------------------------------
elif "🤖 Agente IA" in pagina:
    st.markdown("## 🤖 Agente de Análise de Crédito")
    st.markdown("<small style='color:#7b8099'>Powered by Groq + Llama 3.3 70B</small>", unsafe_allow_html=True)

    # Inicializa histórico
    if "historico" not in st.session_state:
        st.session_state.historico = []

    # Sugestões de perguntas
    st.markdown("<div class='section-title'>Sugestões</div>", unsafe_allow_html=True)
    sugestoes = [
        "Resumo do cenário atual",
        "Período de maior inadimplência",
        "Correlação juros x inadimplência",
        "Impacto da pandemia (2020)",
        "Preveja a inadimplência",
        "Quais features mais importam?",
    ]
    cols_sug = st.columns(3)
    for i, sug in enumerate(sugestoes):
        if cols_sug[i % 3].button(sug, key=f"sug_{i}", use_container_width=True):
            st.session_state.pergunta_rapida = sug

    st.divider()

    # Exibe histórico
    if st.session_state.historico:
        st.markdown("<div class='section-title'>Conversa</div>", unsafe_allow_html=True)
        for msg in st.session_state.historico:
            if msg["role"] == "user":
                st.markdown(f"<div class='chat-user'>👤 {msg['content']}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='chat-bot'>🤖 {msg['content']}</div>", unsafe_allow_html=True)

    # Input da pergunta
    with st.form("form_chat", clear_on_submit=True):
        pergunta = st.text_input(
            "Sua pergunta",
            placeholder="Ex: Como está a inadimplência nos últimos meses?",
            label_visibility="collapsed",
            value=st.session_state.get("pergunta_rapida", "")
        )
        col_btn1, col_btn2 = st.columns([4, 1])
        enviar  = col_btn1.form_submit_button("Enviar →", use_container_width=True)
        limpar  = col_btn2.form_submit_button("Limpar", use_container_width=True)

    if limpar:
        st.session_state.historico = []
        st.session_state.pop("pergunta_rapida", None)
        st.rerun()

    if enviar and pergunta.strip():
        st.session_state.pop("pergunta_rapida", None)

        # Detecta previsão
        palavras_prev = ["preveja", "previsão", "prever", "próximo", "futuro"]
        p = pergunta
        if any(w in p.lower() for w in palavras_prev):
            prev = fazer_previsao()
            ultimo_real = df_bruto['Inadimplência (%)'].iloc[-1]
            p += f"\n\n[Modelo]: previsão={prev:.2f}%, último real={ultimo_real:.2f}%"

        st.session_state.historico.append({"role": "user", "content": pergunta})

        with st.spinner("Analisando..."):
            resposta = chamar_groq(st.session_state.historico)

        st.session_state.historico.append({"role": "assistant", "content": resposta})
        st.rerun()

# -----------------------------------------------------------------------------
# PÁGINA: PREVISÃO
# -----------------------------------------------------------------------------
elif "🔮 Previsão" in pagina:
    st.markdown("## 🔮 Previsão de Inadimplência")

    previsao    = fazer_previsao()
    ultimo_real = df_bruto['Inadimplência (%)'].iloc[-1]
    delta_prev  = previsao - ultimo_real

    col1, col2, col3 = st.columns(3)

    col1.markdown(
        f"<div class='previsao-box'>"
        f"<div style='color:#7b8099;font-size:0.8rem;text-transform:uppercase;letter-spacing:.1em'>Previsão Próximo Mês</div>"
        f"<div style='font-size:2.5rem;font-weight:700;color:#4fc3f7;font-family:DM Mono,monospace;margin:10px 0'>{previsao:.2f}%</div>"
        f"<div style='color:#{'ef5350' if delta_prev > 0 else '66bb6a'};font-size:0.9rem'>{'▲' if delta_prev > 0 else '▼'} {abs(delta_prev):.2f} pp em relação ao último mês</div>"
        f"</div>",
        unsafe_allow_html=True
    )
    col2.markdown(
        f"<div class='metric-card'>"
        f"<div class='metric-label'>Último valor real</div>"
        f"<div class='metric-value'>{ultimo_real:.2f}%</div>"
        f"<div style='color:#7b8099;font-size:0.8rem;margin-top:4px'>{df_bruto.index[-1].strftime('%b/%Y')}</div>"
        f"</div>",
        unsafe_allow_html=True
    )
    col3.markdown(
        f"<div class='metric-card'>"
        f"<div class='metric-label'>Modelo</div>"
        f"<div class='metric-value' style='font-size:1.2rem'>Random Forest</div>"
        f"<div style='color:#7b8099;font-size:0.8rem;margin-top:8px'>R² = 0.556 · MAE = 0.219 pp</div>"
        f"</div>",
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Histórico + Previsão</div>", unsafe_allow_html=True)

    # Gráfico histórico com previsão
    from pandas.tseries.offsets import MonthEnd
    prox_data = df_bruto.index[-1] + pd.DateOffset(months=1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_bruto.index[-36:], y=df_bruto["Inadimplência (%)"].iloc[-36:],
        name="Histórico", line=dict(color="#ef5350", width=2),
        fill='tozeroy', fillcolor='rgba(239,83,80,0.07)'
    ))
    fig.add_trace(go.Scatter(
        x=[df_bruto.index[-1], prox_data],
        y=[ultimo_real, previsao],
        name="Previsão", line=dict(color="#4fc3f7", width=3, dash="dot"),
        mode="lines+markers",
        marker=dict(size=[0, 14], color="#4fc3f7", symbol=["circle", "star"])
    ))
    fig.update_layout(
        paper_bgcolor="#1a1d2e", plot_bgcolor="#1a1d2e",
        font=dict(color="#9099b5", family="DM Sans"),
        yaxis=dict(title="Inadimplência (%)", gridcolor="#2a2d3e"),
        legend=dict(bgcolor="#1a1d2e", bordercolor="#2a2d3e"),
        margin=dict(l=10, r=10, t=10, b=10),
        height=380,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='section-title'>Sobre o Modelo</div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background:#1a1d2e;border:1px solid #2a2d3e;border-radius:12px;padding:20px;line-height:1.8;color:#9099b5'>
    <b style='color:#c5cae9'>Algoritmo:</b> Random Forest Regressor (200 estimadores)<br>
    <b style='color:#c5cae9'>Fonte dos dados:</b> Banco Central do Brasil — API SGS<br>
    <b style='color:#c5cae9'>Features:</b> 24 variáveis (lags 1 e 3 meses, médias móveis 3/6/12 meses, variações mensais e trimestrais)<br>
    <b style='color:#c5cae9'>Divisão:</b> 80% treino (temporal) / 20% teste — sem vazamento de dados<br>
    <b style='color:#c5cae9'>Métricas:</b> R² = 0.556 · RMSE = 0.274 · MAE = 0.219 pp<br>
    <b style='color:#c5cae9'>⚠️ Aviso:</b> Previsão baseada em padrões históricos. Eventos externos podem afetar o resultado.
    </div>
    """, unsafe_allow_html=True)
