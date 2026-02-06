import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

# =========================
# 1. CONFIGURAÇÃO DA PÁGINA
# =========================
st.set_page_config(page_title="LoL Oráculo Pro", page_icon="⚔️", layout="wide")

# =========================
# 2. MOTOR DE DATA SCIENCE
# =========================
@st.cache_data
def load_and_train():
    df = pd.read_csv('Base_M43_Pratique_LOL_RANKED_WIN (1).csv')
    feat = ['blueGoldDiff', 'blueExperienceDiff', 'blueKills', 'blueDeaths', 'blueDragons']
    X = df[feat]
    y = df['blueWins']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Modelos
    log_reg = LogisticRegression(max_iter=1000)
    nb = GaussianNB()
    tree = DecisionTreeClassifier(max_depth=5, random_state=42)

    # Validação cruzada
    scores = {
        "Regressão Logística": cross_val_score(log_reg, X_scaled, y, cv=5, scoring='roc_auc').mean(),
        "Naive Bayes": cross_val_score(nb, X_scaled, y, cv=5, scoring='roc_auc').mean(),
        "Árvore": cross_val_score(tree, X_scaled, y, cv=5, scoring='roc_auc').mean()
    }

    # Treina modelo principal (Logística)
    log_reg.fit(X_scaled, y)

    return df, log_reg, scaler, feat, scores

df_raw, modelo, normalizador, nomes_colunas, scores = load_and_train()

# =========================
# 3. CSS CUSTOMIZADO
# =========================
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #000913 0%, #001524 50%, #062c43 100%) !important;
    }
    [data-testid="stMetric"] {
        background-color: rgba(1, 10, 19, 0.8);
        border: 1px solid #c89b3c;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.5);
    }
    [data-testid="stMetricLabel"] { color: #f0e6d2 !important; font-size: 16px !important; }
    [data-testid="stMetricValue"] { color: #c8aa6e !important; }
    h1, h2, h3 { color: #c89b3c !important; font-family: 'Beaufort for LoL', sans-serif; }
    </style>
    """, unsafe_allow_html=True)

# =========================
# 4. SIDEBAR
# =========================
st.sidebar.title("⚔️ Oráculo LoL")
st.sidebar.markdown("Projeto de Data Science para prever vitórias no LoL Ranked.")
modelo_escolhido = st.sidebar.selectbox("Escolha o Modelo", ["Logística", "Naive Bayes", "Árvore"])
st.sidebar.markdown("### Links")
st.sidebar.markdown("[📂 GitHub](https://github.com/etmelo)")
st.sidebar.markdown("[💼 LinkedIn](https://www.linkedin.com/in/emerson-de-melo-876a8b271/)")

# =========================
# 5. HEADER
# =========================
st.title("⚔️ ORÁCULO DE SUMMONER'S RIFT")
st.markdown("Dashboard interativo para análise e previsão de vitórias no League of Legends Ranked.")
st.markdown("---")

# =========================
# 6. KPIs GLOBAIS
# =========================
m_ouro = df_raw['blueGoldDiff'].mean()
m_xp = df_raw['blueExperienceDiff'].mean()
m_abates = df_raw['blueKills'].mean()
m_winrate = (df_raw['blueWins'].mean()) * 100
melhor_modelo = max(scores, key=scores.get)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Win Rate Azul", f"{m_winrate:.1f}%")
c2.metric("Média Ouro", f"{int(m_ouro)}")
c3.metric("Média XP", f"{int(m_xp)}")
c4.metric("Melhor Modelo (AUC)", f"{scores[melhor_modelo]:.3f}")

st.markdown("---")

# =========================
# 7. PREDIÇÃO INTERATIVA
# =========================
st.header("🎯 Predição Interativa")
col_dados, col_predicao = st.columns([1, 1.5])

with col_dados:
    st.markdown("### Ajuste os Dados da Partida")
    ouro = st.number_input("Diferença de Ouro", value=500, step=100)
    xp = st.number_input("Diferença de XP", value=200, step=100)
    k = st.slider("Abates Azul", 0, 30, 5)
    d = st.slider("Mortes Azul", 0, 30, 4)
    drag = st.radio("Garantiu Dragão?", ["Não", "Sim"], horizontal=True)
    drag_val = 1 if drag == "Sim" else 0

with col_predicao:
    entrada = pd.DataFrame([[ouro, xp, k, d, drag_val]], columns=nomes_colunas)
    entrada_norm = normalizador.transform(entrada)
    prob = modelo.predict_proba(entrada_norm)[0][1]

    if prob > 0.60:
        status_texto = "Vantagem Azul"
        cor_delta = "normal"
    elif prob < 0.40:
        status_texto = "Vantagem Vermelha"
        cor_delta = "inverse"
    else:
        status_texto = "Equilibrado"
        cor_delta = "off"

    st.metric("Status da Previsão", status_texto, delta=f"{prob*100:.1f}%", delta_color=cor_delta)

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={'suffix': "%", 'font': {'color': '#c8aa6e'}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': "#c8aa6e"},
            'bar': {'color': "#0ac8b9"},
            'steps': [
                {'range': [0, 45], 'color': "#5e0b15"},
                {'range': [45, 55], 'color': "#1e2328"},
                {'range': [55, 100], 'color': "#091428"}
            ],
            'threshold': {'line': {'color': "gold", 'width': 4}, 'value': prob * 100}
        }
    ))
    fig_gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', height=380, margin=dict(t=20, b=0))
    st.plotly_chart(fig_gauge, width="stretch", key="gauge")

    if prob > 0.5:
        st.success("TENDÊNCIA: VITÓRIA DO TIME AZUL ✅")
    else:
        st.error("TENDÊNCIA: VITÓRIA DO TIME VERMELHO ❌")

st.markdown("---")

# =========================
# 8. INSIGHTS EXTRAS
# =========================
st.header("📈 Insights Extras")
col_g1, col_g2, col_g3 = st.columns(3)

with col_g1:
    st.subheader("Impacto de Ouro por Vitória")
    fig1 = px.box(df_raw, x="blueWins", y="blueGoldDiff", color="blueWins",
                  color_discrete_map={1: "#0ac8b9", 0: "#5e0b15"})
    fig1.update_layout(paper_bgcolor='rgba(0,0,0,0)', 
                       plot_bgcolor='rgba(0,0,0,0)', 
                       font_color="#f0e6d2")
    st.plotly_chart(fig1, width="stretch", key="boxplot")

with col_g2:
    st.subheader("Importância dos Fatores Técnicos")
    importancia = pd.DataFrame({
        'Fator': ['Ouro', 'XP', 'Abates', 'Mortes', 'Dragão'],
        'Peso': modelo.coef_[0]
    })
    fig2 = px.bar(importancia, x='Peso', y='Fator', orientation='h', color_discrete_sequence=['#c89b3c'])
    fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', 
                       plot_bgcolor='rgba(0,0,0,0)', 
                       font_color="#f0e6d2")
    st.plotly_chart(fig2, width="stretch", key="barplot")

with col_g3:
    st.subheader("Distribuição de Vitórias")
    vit_counts = df_raw['blueWins'].value_counts().rename({0: "Vermelho", 1: "Azul"})
    fig3 = px.pie(values=vit_counts.values, names=vit_counts.index, 
                  color=vit_counts.index, color_discrete_map={"Azul": "#0ac8b9", "Vermelho": "#5e0b15"})
    fig3.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color="#f0e6d2")
    st.plotly_chart(fig3, width="stretch", key="piechart")

st.markdown("---")

# =========================
# 9. TROFÉU DO VENCEDOR
# =========================
st.header("🏆 Resultado Previsto")
if prob > 0.5:
    st.success("🏆 **Time Azul é o favorito!**")
else:
    st.error("🏆 **Time Vermelho é o favorito!**")

# --- RODAPÉ ---
st.markdown("---")
st.markdown("📂 [GitHub](https://github.com/etmelo) | 💼 [LinkedIn](https://www.linkedin.com/in/emerson-de-melo-876a8b271/)")
