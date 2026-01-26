import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
import pickle
from datetime import date
import os

# ============================================
# 1. CONFIGURAÇÃO DA PÁGINA
# ============================================
st.set_page_config(
    page_title="Flight Risk AI",
    page_icon="✈️",
    layout="wide"
)

# Estilização CSS para cards e botões
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin-bottom: 10px;
    }
    .stButton>button {
        width: 100%;
        height: 3em;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================
# 2. CARREGAMENTO DOS ARQUIVOS (CACHE)
# ============================================
@st.cache_resource
def load_resources():
    """Carrega o Modelo e a Lista de Features da pasta local."""
    
    # Verifica existência dos arquivos
    if not os.path.exists('modelo_atrasos_v1.cbm'):
        return None, None
        
    # Carregar Modelo CatBoost
    model = CatBoostClassifier()
    model.load_model('modelo_atrasos_v1.cbm')
    
    # Carregar Lista de Colunas (Features)
    with open('features_list.pkl', 'rb') as f:
        feature_cols = pickle.load(f)
        
    return model, feature_cols

@st.cache_data
def load_data():
    """Carrega o histórico para cálculo de Lags."""
    if not os.path.exists('historico_voos.csv'):
        return None
        
    df = pd.read_csv('historico_voos.csv')
    df['reporting_period'] = pd.to_datetime(df['reporting_period'])
    return df

# Executa carregamento
model, feature_cols = load_resources()
df_history = load_data()

# Tratamento de erro se arquivos não existirem
if model is None or df_history is None:
    st.error("""
    ❌ **Arquivos não encontrados!**
    
    Certifique-se de que estes 3 arquivos estão na mesma pasta do 'app.py':
    1. `modelo_atrasos_v1.cbm`
    2. `features_list.pkl`
    3. `historico_voos.csv`
    """)
    st.stop()

# ============================================
# 3. BARRA LATERAL (INPUTS)
# ============================================
st.sidebar.header("🛠️ Configuração do Voo")
st.sidebar.info("Defina os parâmetros para simulação.")

# Inputs
empresas = sorted(df_history['airline_name'].unique())
airline_input = st.sidebar.selectbox("Companhia Aérea", empresas)

# Filtro dinâmico de rotas
rotas_da_empresa = sorted(df_history[df_history['airline_name'] == airline_input]['flight_route'].unique())
if not rotas_da_empresa:
    rotas_da_empresa = ["Sem rotas disponíveis"]

route_input = st.sidebar.selectbox("Rota de Voo", rotas_da_empresa)
date_input = st.sidebar.date_input("Data Prevista", value=date(2025, 12, 1))

# Volume de voos (sugestão baseada na média histórica)
media_voos_historica = 50 
if route_input != "Sem rotas disponíveis":
    subset = df_history[(df_history['airline_name'] == airline_input) & (df_history['flight_route'] == route_input)]
    if not subset.empty:
        media_voos_historica = int(subset['total_flights'].mean())

flights_input = st.sidebar.number_input("Volume de Voos", min_value=1, value=media_voos_historica)

# ============================================
# 4. PREPARAÇÃO DE DADOS (ENGINEERING)
# ============================================
def prepare_input_data(airline, route, date_val, flights, history_df, feature_list):
    """Reconstrói as features (Lags, Sazonalidade) usando o histórico."""
    
    # 1. Busca histórico recente
    history_subset = history_df[
        (history_df['airline_name'] == airline) & 
        (history_df['flight_route'] == route)
    ].sort_values('reporting_period', ascending=False)
    
    # 2. Base do input
    input_data = {
        'airline_name': airline,
        'flight_route': route,
        'total_flights': flights,
        'origin_destination_country': history_subset.iloc[0]['origin_destination_country'] if not history_subset.empty else "Unknown"
    }
    
    # 3. Features Temporais
    month = date_val.month
    input_data['feat_month_sin'] = np.sin(2 * np.pi * month / 12)
    input_data['feat_month_cos'] = np.cos(2 * np.pi * month / 12)
    input_data['feat_traffic_log'] = np.log1p(flights)
    
    # 4. Features de LAG
    if not history_subset.empty:
        last_record = history_subset.iloc[0]
        # Risco do último mês conhecido
        last_risk = (last_record['number_flights_delayed'] + last_record['number_flights_cancelled']) / last_record['total_flights']
        
        input_data['feat_lag_1m'] = last_risk
        input_data['feat_roll_3m'] = last_risk
        input_data['feat_roll_std_3m'] = 0.05 
        input_data['feat_lag_12m'] = last_risk
        input_data['feat_airline_global_risk_lag1'] = last_record.get('feat_airline_global_risk_lag1', 0.1)
        input_data['feat_route_global_risk_lag1'] = last_record.get('feat_route_global_risk_lag1', 0.1)
    else:
        # Cold Start
        for col in ['feat_lag_1m', 'feat_roll_3m', 'feat_roll_std_3m', 'feat_lag_12m', 
                    'feat_airline_global_risk_lag1', 'feat_route_global_risk_lag1']:
            input_data[col] = -1

    df_input = pd.DataFrame([input_data])
    
    # Retorna apenas as colunas esperadas pelo modelo
    return df_input[feature_list]

# ============================================
# 5. EXECUÇÃO E EXIBIÇÃO
# ============================================
st.title("✈️ Predictor de Risco de Voo")
st.markdown(f"**Análise para:** {airline_input} | **Rota:** {route_input}")

# Contexto Histórico
with st.expander("📊 Ver Histórico Recente", expanded=True):
    if route_input != "Sem rotas disponíveis":
        hist_rec = df_history[
            (df_history['airline_name'] == airline_input) & 
            (df_history['flight_route'] == route_input)
        ].sort_values('reporting_period', ascending=False).head(1)
        
        if not hist_rec.empty:
            last_delay = (hist_rec.iloc[0]['number_flights_delayed'] / hist_rec.iloc[0]['total_flights'])
            c1, c2, c3 = st.columns(3)
            c1.metric("Última Taxa de Atraso", f"{last_delay:.1%}")
            c2.metric("Última Data", hist_rec.iloc[0]['reporting_period'].strftime('%Y-%m'))
            c3.metric("Total Voos", int(hist_rec.iloc[0]['total_flights']))
        else:
            st.warning("Sem dados históricos para esta rota.")

st.divider()

# Botão Calcular
col_btn, col_empty = st.columns([1, 2])
with col_btn:
    calcular = st.button("CALCULAR RISCO FUTURO", type="primary")

if calcular:
    if route_input == "Sem rotas disponíveis":
        st.error("Configuração inválida.")
    else:
        with st.spinner('Processando modelo...'):
            # 1. Prepara dados
            X_pred = prepare_input_data(airline_input, route_input, date_input, flights_input, df_history, feature_cols)
            
            # 2. Predição
            probs = model.predict_proba(X_pred)[0]
            
            # --- CORREÇÃO DO ERRO ---
            # Convertemos explicitamente para int para usar como chave do dicionário
            pred_class = int(model.predict(X_pred)[0]) 
            
            # 3. Exibição
            st.subheader("Resultado da Previsão")
            
            labels = {0: "BAIXO RISCO", 1: "RISCO MÉDIO", 2: "ALTO RISCO"}
            colors = {0: "#28a745", 1: "#ffc107", 2: "#dc3545"} 
            descricoes = {
                0: "Operação prevista dentro da normalidade.",
                1: "Atenção: Chance considerável de atrasos parciais.",
                2: "ALERTA CRÍTICO: Alta probabilidade de atrasos severos."
            }
            
            cols = st.columns([1, 1.5])
            
            with cols[0]:
                st.markdown(f"""
                <div style="background-color: {colors[pred_class]}; padding: 30px; border-radius: 15px; color: white; text-align: center; box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);">
                    <h1 style="margin:0; font-size: 2.5em;">{labels[pred_class]}</h1>
                    <p style="margin-top:10px; font-size: 1.1em;">Status Previsto</p>
                </div>
                """, unsafe_allow_html=True)
                
            with cols[1]:
                st.write("**Probabilidades Detalhadas:**")
                st.write(f"🟢 Baixo: {probs[0]:.1%}")
                st.progress(float(probs[0]))
                st.write(f"🟡 Médio: {probs[1]:.1%}")
                st.progress(float(probs[1]))
                st.write(f"🔴 Alto: {probs[2]:.1%}")
                st.progress(float(probs[2]))

            st.info(f"💡 **Interpretação:** {descricoes[pred_class]}")

else:
    st.markdown("👈 *Configure os parâmetros e clique em calcular.*")