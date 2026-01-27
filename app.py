import streamlit as st
import pandas as pd
import numpy as np
import pickle
from catboost import CatBoostRegressor
import plotly.graph_objects as go
from datetime import datetime
import google.generativeai as genai

# ============================================
# 1. CONFIGURATION & STYLES
# ============================================
st.set_page_config(page_title="Flight Risk AI", page_icon="✈️", layout="wide")

st.markdown("""
    <style>
    .stMetric { background-color: #f8f9fa; border: 1px solid #dee2e6; border-radius: 5px; }
    .stButton>button { width: 100%; font-weight: bold; border-radius: 5px; }
    </style>
    """, unsafe_allow_html=True)

# ============================================
# 2. LOAD DATA
# ============================================
@st.cache_resource
def load_artifacts():
    try:
        with open('app_context.pkl', 'rb') as f:
            ctx = pickle.load(f)
        
        model_delay = CatBoostRegressor()
        model_delay.load_model('model_delay.cbm')
        
        model_cancel = CatBoostRegressor()
        model_cancel.load_model('model_cancel.cbm')
        
        return model_delay, model_cancel, ctx
    except FileNotFoundError:
        return None, None, None

@st.cache_data
def load_history():
    try:
        df = pd.read_pickle("seasonal_route_stats.pkl")
        # Ensure correct types for filtering
        df['airline_name'] = df['airline_name'].astype(str)
        df['flight_route'] = df['flight_route'].astype(str)
        return df
    except FileNotFoundError:
        return pd.DataFrame()

model_delay, model_cancel, ctx = load_artifacts()
df_stats = load_history()

if not ctx or df_stats.empty:
    st.error("❌ Error: Files missing or empty. Please check your .pkl and .cbm files.")
    st.stop()

# ============================================
# 3. HELPER: ESTIMATION LOGIC
# ============================================
def get_operational_estimates(airline, route, month):
    """
    Estimates operational inputs based on HISTORICAL averages.
    Ignores empty future rows (2026) by dropping NaNs.
    """
    SAFE_VOL = 1.0
    SAFE_RISK = 0.5
    
    # Filter for this specific airline and route
    route_data = df_stats[
        (df_stats['airline_name'] == airline) & 
        (df_stats['flight_route'] == route)
    ]
    
    if route_data.empty:
        return SAFE_VOL, SAFE_RISK, "No history for this route."

    # Try 1: Specific Month Average (e.g., Average of May 2023, May 2024)
    month_data = route_data[route_data['month'] == month]
    
    # CRITICAL: Drop rows where data is missing (future flights)
    valid_month_data = month_data.dropna(subset=['avg_vol_vs_avg', 'avg_severity'])
    
    if not valid_month_data.empty:
        est_vol = valid_month_data['avg_vol_vs_avg'].mean()
        est_risk = valid_month_data['avg_severity'].mean()
        month_name = datetime(2000, month, 1).strftime('%B')
        return est_vol, est_risk, f"Based on historical {month_name} averages."

    # Try 2: Annual Average (Fallback if no history for that specific month)
    # Uses all valid data for this route across all months
    valid_route_data = route_data.dropna(subset=['avg_vol_vs_avg', 'avg_severity'])
    
    if not valid_route_data.empty:
        est_vol = valid_route_data['avg_vol_vs_avg'].mean()
        est_risk = valid_route_data['avg_severity'].mean()
        return est_vol, est_risk, "No May data; estimated from annual route average."

    return SAFE_VOL, SAFE_RISK, "New Route (Using Standard Defaults)"

# ============================================
# 4. SIDEBAR - DYNAMIC INPUTS
# ============================================
st.sidebar.header("✈️ Flight Parameters")

# --- SECRETS HANDLING ---
try:
    api_key = st.secrets["GEMINI_API_KEY"]
except (FileNotFoundError, KeyError):
    # Fallback for local testing without secrets.toml
    api_key = st.sidebar.text_input("Gemini API Key", type="password")

# --- 1. AIRLINE SELECTION ---
# Get unique airlines from the stats file to ensure they have data
available_airlines = sorted(df_stats['airline_name'].unique())
airline = st.sidebar.selectbox("Airline", available_airlines)

# --- 2. ROUTE SELECTION (FILTERED) ---
# Filter routes to show ONLY those flown by the selected airline
available_routes = sorted(df_stats[df_stats['airline_name'] == airline]['flight_route'].unique())
route = st.sidebar.selectbox("Route", available_routes)

# --- 3. DATE SELECTION ---
travel_date = st.sidebar.date_input("Travel Date", value=datetime.now())
month = travel_date.month

# --- 4. AUTO-ESTIMATION ---
est_vol, est_risk, context_msg = get_operational_estimates(airline, route, month)

st.sidebar.markdown("---")
st.sidebar.subheader("📡 Operational Estimation")
st.sidebar.caption(f"ℹ️ {context_msg}")

with st.sidebar.expander("⚙️ Adjust Estimates", expanded=True):
    vol_val = st.slider("Congestion Est.", 0.5, 2.0, float(est_vol), format="%.2f")
    recent_perf = st.slider("Momentum Est.", 0.0, 5.0, float(est_risk), format="%.2f")

# ============================================
# 5. PREDICTION ENGINE
# ============================================
def make_prediction():
    # 1. Defaults
    input_data = ctx['defaults'].copy()
    
    # 2. Inject User Inputs
    input_data['airline_name'] = airline
    input_data['flight_route'] = route
    input_data['route_month_id'] = f"{route}_{month}"
    
    # Time Features
    input_data['month_sin'] = np.sin(2 * np.pi * month / 12)
    input_data['month_cos'] = np.cos(2 * np.pi * month / 12)
    input_data['is_winter_holiday'] = 1 if month in [12, 1] else 0
    input_data['is_summer_holiday'] = 1 if month in [7, 8] else 0
    
    # Operational Estimates
    input_data['vol_vs_avg'] = vol_val
    input_data['severity_score_lag_1'] = recent_perf
    input_data['severity_score_roll_mean_3m'] = recent_perf 
    
    # 3. Create DataFrame
    df_input = pd.DataFrame([input_data])
    
    # --- CRITICAL FIX: Ensure EXACT Column Match ---
    # Loop through the columns the model expects (from your context file)
    expected_cols = ctx['feature_columns']
    
    for col in expected_cols:
        if col not in df_input.columns:
            # If a column is missing, fill it with 0 to prevent crash
            df_input[col] = 0 
            
    # Reorder columns to match training order strictly
    df_input = df_input[expected_cols]
    # -----------------------------------------------

    # 4. Predict
    delay_log = model_delay.predict(df_input)[0]
    cancel_log = model_cancel.predict(df_input)[0]
    
    return max(0, np.expm1(delay_log)), max(0, np.expm1(cancel_log))

# ============================================
# 6. MAIN DASHBOARD
# ============================================
st.title("🛫 Flight Risk AI")

# Session State for Persistence
if "prediction" not in st.session_state:
    st.session_state.prediction = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if st.button("CALCULATE RISK", type="primary"):
    sev, canc_rate = make_prediction()
    
    # Logic for labels
    if sev < 0.5: lbl, col = "LOW RISK", "green"
    elif sev < 2.0: lbl, col = "MEDIUM RISK", "orange"
    else: lbl, col = "HIGH RISK", "red"
    
    st.session_state.prediction = {
        "severity": sev, "cancel_rate": canc_rate,
        "label": lbl, "color": col,
        "airline": airline, "route": route, "date": str(travel_date)
    }
    st.session_state.chat_history = [] # Reset chat on new prediction

# Display Results
if st.session_state.prediction:
    res = st.session_state.prediction
    
    st.divider()
    c1, c2 = st.columns([1, 1])
    
    with c1:
        st.subheader("Delay Intensity")
        st.markdown(f"<h2 style='color:{res['color']}'>{res['label']} ({res['severity']:.2f})</h2>", unsafe_allow_html=True)
        fig = go.Figure(go.Indicator(
            mode = "gauge+number", value = min(res['severity'], 5.0),
            gauge = {'axis': {'range': [0, 5]}, 'bar': {'color': res['color']}}
        ))
        fig.update_layout(height=200, margin=dict(t=0,b=0,l=20,r=20))
        st.plotly_chart(fig, use_container_width=True)
        
    with c2:
        st.subheader("Cancellation Chance")
        pct = res['cancel_rate'] * 100
        st.metric("Probability", f"{pct:.1f}%")
        st.progress(min(pct/5, 1.0))
        st.caption(f"Estimated using congestion: {vol_val:.2f}")

    # ============================================
    # 7. AI ASSISTANT
    # ============================================
    st.divider()
    st.subheader("🤖 Travel Assistant")
    
    for msg in st.session_state.chat_history:
        st.chat_message(msg["role"]).write(msg["content"])
        
    if prompt := st.chat_input("Ask about backup plans..."):
        if not api_key:
            st.error("⚠️ Please provide a Gemini API Key.")
        else:
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)
            
            context = f"""
            Role: Expert Travel Agent.
            Flight: {res['airline']} {res['route']} on {res['date']}.
            Risk Assessment:
            - Delay Severity: {res['severity']:.2f}/5.0 (High if > 2.0)
            - Cancellation Probability: {res['cancel_rate']*100:.1f}%
            User Question: "{prompt}"
            Provide actionable advice.
            """
            
            try:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel("gemini-1.5-flash-002")
                response = model.generate_content(context)
                bot_reply = response.text
                
                st.session_state.chat_history.append({"role": "assistant", "content": bot_reply})
                st.chat_message("assistant").write(bot_reply)
            except Exception as e:
                st.error(f"AI Error: {e}")