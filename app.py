import streamlit as st
import pandas as pd
import numpy as np
import pickle
from catboost import CatBoostRegressor
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random

# Import airport data module
from data.airport_data import get_airport_options, parse_airport_selection, get_route_code

# ============================================
# 1. CONFIGURATION & STYLES
# ============================================
st.set_page_config(page_title="Flight Risk AI", page_icon="✈️", layout="wide")

st.markdown("""
    <style>
    .stMetric { background-color: #f8f9fa; border: 1px solid #dee2e6; border-radius: 5px; }
    .stButton>button { width: 100%; font-weight: bold; border-radius: 5px; }

    /* Container styling for flight cards */
    div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"] {
        background: #ffffff;
        border-radius: 12px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border: 1px solid #e0e0e0;
    }

    /* Dark theme support */
    @media (prefers-color-scheme: dark) {
        div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"] {
            background: #1e1e1e;
            border: 1px solid #333;
            box-shadow: 0 2px 8px rgba(255,255,255,0.05);
        }
    }

    /* Risk bar */
    .risk-bar {
        height: 8px;
        width: 100%;
        margin: -15px -15px 15px -15px;
        border-radius: 12px 12px 0 0;
    }
    .risk-bar-low {
        background: linear-gradient(90deg, #28a745 0%, #38ef7d 100%);
    }
    .risk-bar-medium {
        background: linear-gradient(90deg, #ffc107 0%, #ff9800 100%);
    }
    .risk-bar-high {
        background: linear-gradient(90deg, #dc3545 0%, #ff5252 100%);
    }

    /* Badges */
    .risk-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 15px;
        font-weight: bold;
        font-size: 11px;
    }
    .risk-low { background-color: #d4edda; color: #155724; }
    .risk-medium { background-color: #fff3cd; color: #856404; }
    .risk-high { background-color: #f8d7da; color: #721c24; }

    .recommended-badge {
        background-color: #ffd700;
        color: #333;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: bold;
        display: inline-block;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================
# 2. LOAD API KEY FROM SECRETS (Task 1)
# ============================================
try:
    # Try Streamlit secrets first
    api_key = st.secrets["GEMINI_API_KEY"]
except (FileNotFoundError, KeyError):
    try:
        # Fallback: read directly from secrets.toml
        import toml
        with open('secrets.toml', 'r') as f:
            secrets = toml.load(f)
        api_key = secrets['GEMINI_API_KEY']
        if api_key:
            st.success("✅ Gemini API Key loaded successfully!")
    except (FileNotFoundError, KeyError, Exception):
        api_key = None
        st.warning("⚠️ Gemini API Key not found in secrets.toml. AI Assistant will be disabled.")

# ============================================
# 3. LOAD DATA
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
# 4. HELPER FUNCTIONS
# ============================================
def get_operational_estimates(airline, route, month):
    """Estimates operational inputs based on HISTORICAL averages."""
    SAFE_VOL = 1.0
    SAFE_RISK = 0.5

    route_data = df_stats[
        (df_stats['airline_name'] == airline) &
        (df_stats['flight_route'] == route)
    ]

    if route_data.empty:
        return SAFE_VOL, SAFE_RISK, "No history for this route."

    month_data = route_data[route_data['month'] == month]
    valid_month_data = month_data.dropna(subset=['avg_vol_vs_avg', 'avg_severity'])

    if not valid_month_data.empty:
        est_vol = valid_month_data['avg_vol_vs_avg'].mean()
        est_risk = valid_month_data['avg_severity'].mean()
        month_name = datetime(2000, month, 1).strftime('%B')
        return est_vol, est_risk, f"Based on historical {month_name} averages."

    valid_route_data = route_data.dropna(subset=['avg_vol_vs_avg', 'avg_severity'])

    if not valid_route_data.empty:
        est_vol = valid_route_data['avg_vol_vs_avg'].mean()
        est_risk = valid_route_data['avg_severity'].mean()
        return est_vol, est_risk, "Estimated from annual route average."

    return SAFE_VOL, SAFE_RISK, "New Route (Using Standard Defaults)"


def make_prediction(airline, route, month, vol_val, recent_perf):
    """Generate prediction for a specific flight."""
    input_data = ctx['defaults'].copy()

    input_data['airline_name'] = airline
    input_data['flight_route'] = route
    input_data['route_month_id'] = f"{route}_{month}"

    input_data['month_sin'] = np.sin(2 * np.pi * month / 12)
    input_data['month_cos'] = np.cos(2 * np.pi * month / 12)
    input_data['is_winter_holiday'] = 1 if month in [12, 1] else 0
    input_data['is_summer_holiday'] = 1 if month in [7, 8] else 0

    input_data['vol_vs_avg'] = vol_val
    input_data['severity_score_lag_1'] = recent_perf
    input_data['severity_score_roll_mean_3m'] = recent_perf

    df_input = pd.DataFrame([input_data])
    expected_cols = ctx['feature_columns']

    for col in expected_cols:
        if col not in df_input.columns:
            df_input[col] = 0

    df_input = df_input[expected_cols]

    delay_log = model_delay.predict(df_input)[0]
    cancel_log = model_cancel.predict(df_input)[0]

    return max(0, np.expm1(delay_log)), max(0, np.expm1(cancel_log))


def generate_flight_options(origin_code, dest_code, base_date, airline, vol_val, recent_perf):
    """Generate flight options for -1, 0, +1 days with risk predictions."""
    flights = []
    route = f"{origin_code}-{dest_code}"

    # Generate flights for 3 days (-1, 0, +1)
    for day_offset in [-1, 0, 1]:
        flight_date = base_date + timedelta(days=day_offset)
        month = flight_date.month

        # Generate 2-3 flights per day with different times
        flight_times = [
            ("06:30", "08:45"),
            ("10:15", "12:30"),
            ("14:00", "16:15"),
            ("18:30", "20:45"),
            ("21:00", "23:15")
        ]

        # Select random subset of times for variety
        selected_times = random.sample(flight_times, min(3, len(flight_times)))

        for dep_time, arr_time in selected_times:
            sev, canc_rate = make_prediction(airline, route, month, vol_val, recent_perf)

            # Add some variation to make flights more realistic
            sev_variation = sev * (0.9 + random.random() * 0.2)
            canc_variation = canc_rate * (0.9 + random.random() * 0.2)

            if sev_variation < 0.5:
                risk_label = "LOW RISK"
                risk_color = "green"
                risk_class = "low"
            elif sev_variation < 2.0:
                risk_label = "MEDIUM RISK"
                risk_color = "orange"
                risk_class = "medium"
            else:
                risk_label = "HIGH RISK"
                risk_color = "red"
                risk_class = "high"

            flights.append({
                "date": flight_date,
                "day_offset": day_offset,
                "departure_time": dep_time,
                "arrival_time": arr_time,
                "origin": origin_code,
                "destination": dest_code,
                "airline": airline,
                "route": route,
                "severity": sev_variation,
                "cancel_rate": canc_variation,
                "risk_label": risk_label,
                "risk_color": risk_color,
                "risk_class": risk_class,
                "flight_number": f"{airline[:2].upper()}{random.randint(1000, 9999)}"
            })

    # Sort by severity (lowest risk first)
    flights.sort(key=lambda x: x['severity'])

    return flights


def get_ai_recommendation(flights, travel_purpose, has_commitment, commitment_time, commitment_priority, origin, destination, travel_date):
    """Get AI recommendation for best flight based on user preferences."""
    if not api_key:
        return None

    # Build context for AI
    flights_summary = "\n".join([
        f"- Flight {f['flight_number']}: {f['date'].strftime('%Y-%m-%d')} {f['departure_time']}-{f['arrival_time']}, "
        f"Risk: {f['risk_label']} (Severity: {f['severity']:.2f}, Cancellation: {f['cancel_rate']*100:.1f}%)"
        for f in flights[:6]  # Top 6 flights
    ])

    commitment_info = ""
    if has_commitment:
        commitment_info = f"""
        - Has commitment after arrival: Yes
        - Commitment time: {commitment_time}
        - Priority: {commitment_priority}
        """
    else:
        commitment_info = "- Has commitment after arrival: No"

    context = f"""
    You are an expert travel advisor. Analyze the following flight options and recommend the best choices
    based on the traveler's preferences. Be concise and actionable.

    TRAVELER PROFILE:
    - Travel purpose: {travel_purpose}
    {commitment_info}
    - Route: {origin} to {destination}
    - Preferred date: {travel_date.strftime('%Y-%m-%d')}

    AVAILABLE FLIGHTS (sorted by risk, lowest first):
    {flights_summary}

    Please provide:
    1. Your TOP RECOMMENDATION with reasoning (considering their commitment and travel purpose)
    2. A BACKUP OPTION in case of issues
    3. Any specific advice based on their travel purpose and commitment status

    Keep your response brief and focused on actionable advice.
    """

    # Use a compatibility layer: prefer `google.genai`, fallback to `google.generativeai`
    try:
        # Lazy import to avoid heavy startup cost
        from google import genai as new_genai

        # Instantiate client (api_key if available)
        client = None
        try:
            client = new_genai.Client(api_key=api_key) if api_key else new_genai.Client()
        except Exception:
            # Some versions accept configure/global setup
            try:
                new_genai.configure(api_key=api_key)
                client = new_genai
            except Exception:
                client = new_genai.Client()

        # google.genai uses client.models.generate_content
        if hasattr(client, 'models') and hasattr(client.models, 'generate_content'):
            resp = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=context
            )
            return resp.text
        else:
            raise Exception("Client does not support models.generate_content, trying fallback")

    except Exception:
        # Fallback to older package if available
        try:
            import google.generativeai as old_genai
            old_genai.configure(api_key=api_key)
            model = old_genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(context)
            return response.text
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "quota" in error_msg.lower():
                return "🚫 AI Assistant temporarily unavailable (quota exceeded). Please try again later."
            elif "404" in error_msg:
                return "⚙️ AI model temporarily unavailable. Please try again later."
            else:
                return f"Could not generate recommendation: {error_msg}"


# ============================================
# 5. SIDEBAR - USER INPUTS
# ============================================
st.sidebar.header("✈️ Flight Search")

# --- AIRPORT SELECTION (Task 3) ---
st.sidebar.subheader("📍 Route")
airport_options = get_airport_options()

origin_selection = st.sidebar.selectbox(
    "Origin",
    options=airport_options,
    index=0,
    help="Select departure city or airport"
)

# Filter destination to exclude origin
dest_options = [opt for opt in airport_options if opt != origin_selection]
destination_selection = st.sidebar.selectbox(
    "Destination",
    options=dest_options,
    index=0 if dest_options else None,
    help="Select arrival city or airport"
)

# Parse selections
origin_data = parse_airport_selection(origin_selection)
dest_data = parse_airport_selection(destination_selection)
origin_code = origin_data['airport_code']
dest_code = dest_data['airport_code']

# --- DATE SELECTION ---
st.sidebar.subheader("📅 Travel Date")
travel_date = st.sidebar.date_input("Departure Date", value=datetime.now() + timedelta(days=7))
month = travel_date.month

# --- AIRLINE SELECTION ---
available_airlines = sorted(df_stats['airline_name'].unique())
airline = st.sidebar.selectbox("Preferred Airline", available_airlines)

# --- TRAVEL PURPOSE (Task 4) ---
st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Travel Purpose")

travel_purposes = ["Business", "Tourism", "Family Visit", "Medical", "Education", "Other"]
travel_purpose = st.sidebar.selectbox("Purpose of Travel", travel_purposes)

has_commitment = st.sidebar.checkbox("I have a commitment after arrival", value=False)

commitment_time = None
commitment_priority = None

if has_commitment:
    commitment_time = st.sidebar.time_input(
        "Commitment Time",
        value=datetime.strptime("14:00", "%H:%M").time()
    )

    priority_options = [
        "Cannot be late - Critical meeting",
        "Prefer not to be late - Important",
        "Some flexibility - Can be slightly late",
        "Very flexible - Can reschedule if needed"
    ]
    commitment_priority = st.sidebar.selectbox("Commitment Priority", priority_options)

# --- OPERATIONAL ESTIMATES ---
st.sidebar.markdown("---")
st.sidebar.subheader("📡 Operational Estimates")

# Generate route for estimation
route_for_estimate = f"{origin_code}-{dest_code}"
est_vol, est_risk, context_msg = get_operational_estimates(airline, route_for_estimate, month)
st.sidebar.caption(f"ℹ️ {context_msg}")

with st.sidebar.expander("⚙️ Adjust Estimates", expanded=False):
    vol_val = st.slider("Congestion Level", 0.5, 2.0, float(est_vol), format="%.2f")
    recent_perf = st.slider("Recent Performance", 0.0, 5.0, float(est_risk), format="%.2f")

# ============================================
# 6. MAIN DASHBOARD
# ============================================
st.title("🛫 Flight Risk AI")
st.markdown(f"**Route:** {origin_data['city_name']} ({origin_code}) → {dest_data['city_name']} ({dest_code})")

# Session State
if "flights" not in st.session_state:
    st.session_state.flights = None
if "ai_recommendation" not in st.session_state:
    st.session_state.ai_recommendation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "selected_flight" not in st.session_state:
    st.session_state.selected_flight = None

# --- FIND FLIGHTS BUTTON (Task 5) ---
if st.button("🔍 FIND FLIGHTS", type="primary", use_container_width=True):
    with st.spinner("Searching for flights and analyzing risks..."):
        # Generate flight options
        flights = generate_flight_options(
            origin_code, dest_code, travel_date,
            airline, vol_val, recent_perf
        )
        st.session_state.flights = flights

        # Get AI recommendation (Task 6)
        if api_key:
            recommendation = get_ai_recommendation(
                flights, travel_purpose, has_commitment,
                commitment_time, commitment_priority,
                f"{origin_data['city_name']} ({origin_code})",
                f"{dest_data['city_name']} ({dest_code})",
                travel_date
            )
            st.session_state.ai_recommendation = recommendation

        st.session_state.chat_history = []

# --- DISPLAY FLIGHTS (Task 5) ---
if st.session_state.flights:
    flights = st.session_state.flights

    # AI Recommendation Section (Task 6)
    if st.session_state.ai_recommendation:
        st.divider()
        st.subheader("🤖 AI Travel Advisor Recommendation")
        with st.expander("View personalized recommendation", expanded=True):
            st.markdown(st.session_state.ai_recommendation)

    st.divider()

    # Group flights by date
    dates = sorted(set(f['date'] for f in flights))

    # Create tabs for each day
    day_labels = []
    for d in dates:
        offset = (d - travel_date).days
        if offset == -1:
            day_labels.append(f"📅 Day Before ({d.strftime('%b %d')})")
        elif offset == 0:
            day_labels.append(f"⭐ Selected Date ({d.strftime('%b %d')})")
        else:
            day_labels.append(f"📅 Day After ({d.strftime('%b %d')})")

    tabs = st.tabs(day_labels)

    for tab, date in zip(tabs, dates):
        with tab:
            day_flights = [f for f in flights if f['date'] == date]
            day_flights.sort(key=lambda x: x['severity'])

            for i, flight in enumerate(day_flights):
                # Determine if this is recommended (lowest risk for the selected date)
                is_recommended = (i == 0 and flight['day_offset'] == 0)

                # Create flight card usando componentes nativos do Streamlit
                with st.container():
                    # Risk bar at top
                    st.markdown(
                        f'<div class="risk-bar risk-bar-{flight["risk_class"]}"></div>',
                        unsafe_allow_html=True
                    )

                    # Card content
                    col_header, col_badge = st.columns([4, 1])
                    with col_header:
                        st.markdown(f"### ✈️ {flight['flight_number']}")
                    with col_badge:
                        if is_recommended:
                            st.markdown('<span class="recommended-badge">⭐ RECOMENDADO</span>', unsafe_allow_html=True)

                    # Flight details
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**🛫 Partida:** {flight['departure_time']}")
                        st.markdown(f"**🛬 Chegada:** {flight['arrival_time']}")
                    with col2:
                        st.markdown(f"**Rota:** {flight['origin']} → {flight['destination']}")
                        st.markdown(f"**Companhia:** {flight['airline']}")

                    # Risk and stats footer
                    col_risk, col_stats = st.columns([1, 2])
                    with col_risk:
                        st.markdown(
                            f'<span class="risk-badge risk-{flight["risk_class"]}">{flight["risk_label"]}</span>',
                            unsafe_allow_html=True
                        )
                    with col_stats:
                        st.markdown(f"**Score Atraso:** {flight['severity']:.2f}")
                        st.markdown(f"**Chance Cancelamento:** {flight['cancel_rate']*100:.1f}%")

                    # Button for selection
                    if st.button(f"✈️ Selecionar Voo {flight['flight_number']}",
                               key=f"select_{flight['flight_number']}_{date}",
                               use_container_width=True):
                        st.session_state.selected_flight = flight
                        st.success(f"Voo {flight['flight_number']} selecionado!")
                        st.rerun()
    if st.session_state.selected_flight:
        st.subheader("📋 Selected Flight Details")
        flight = st.session_state.selected_flight

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Flight Information")
            st.markdown(f"**Flight Number:** {flight['flight_number']}")
            st.markdown(f"**Date:** {flight['date'].strftime('%A, %B %d, %Y')}")
            st.markdown(f"**Departure:** {flight['departure_time']} from {flight['origin']}")
            st.markdown(f"**Arrival:** {flight['arrival_time']} at {flight['destination']}")
            st.markdown(f"**Airline:** {flight['airline']}")

        with col2:
            st.markdown("### Risk Assessment")

            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=min(flight['severity'], 5.0),
                title={'text': "Delay Intensity"},
                gauge={
                    'axis': {'range': [0, 5]},
                    'bar': {'color': flight['risk_color']},
                    'steps': [
                        {'range': [0, 0.5], 'color': "lightgreen"},
                        {'range': [0.5, 2], 'color': "lightyellow"},
                        {'range': [2, 5], 'color': "lightcoral"}
                    ]
                }
            ))
            fig.update_layout(height=200, margin=dict(t=50, b=0, l=20, r=20))
            st.plotly_chart(fig, use_container_width=True)

            st.metric("Cancellation Probability", f"{flight['cancel_rate']*100:.1f}%")

    # ============================================
    # 7. AI TRAVEL ASSISTANT CHAT (Task 6)
    # ============================================
    st.divider()
    st.subheader("💬 Travel Assistant")
    st.caption("Ask questions about your flight options, backup plans, or travel advice")

    for msg in st.session_state.chat_history:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("Ask about backup plans, connections, or travel tips..."):
        if not api_key:
            st.error("⚠️ Gemini API Key not configured in secrets.toml")
        else:
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)

            # Build context with all flight info and user preferences
            selected_info = ""
            if st.session_state.selected_flight:
                f = st.session_state.selected_flight
                selected_info = f"""
                Selected Flight: {f['flight_number']} on {f['date'].strftime('%Y-%m-%d')}
                - Departure: {f['departure_time']} from {f['origin']}
                - Arrival: {f['arrival_time']} at {f['destination']}
                - Risk Level: {f['risk_label']} (Severity: {f['severity']:.2f})
                """

            commitment_info = ""
            if has_commitment:
                commitment_info = f"""
                Traveler has a commitment at {commitment_time}
                Priority: {commitment_priority}
                """

            context = f"""
            Role: Expert Travel Agent Assistant.

            ROUTE: {origin_data['city_name']} ({origin_code}) to {dest_data['city_name']} ({dest_code})
            TRAVEL DATE: {travel_date.strftime('%Y-%m-%d')}
            TRAVEL PURPOSE: {travel_purpose}
            {commitment_info}

            {selected_info}

            Available Flights Summary:
            {chr(10).join([f"- {f['flight_number']}: {f['date'].strftime('%m/%d')} {f['departure_time']}, Risk: {f['risk_label']}" for f in flights[:5]])}

            User Question: "{prompt}"

            Provide helpful, actionable advice. Be concise.
            """

            # Use same compatibility approach as recommendations: prefer google.genai, fallback to google.generativeai
            try:
                # Lazy import new client
                from google import genai as new_genai

                try:
                    client = new_genai.Client(api_key=api_key) if api_key else new_genai.Client()
                except Exception:
                    try:
                        new_genai.configure(api_key=api_key)
                        client = new_genai
                    except Exception:
                        client = new_genai.Client()

                # google.genai uses client.models.generate_content
                if hasattr(client, 'models') and hasattr(client.models, 'generate_content'):
                    resp = client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=context
                    )
                    bot_reply = resp.text
                else:
                    # Fallback to old library
                    raise Exception("Client does not support models.generate_content, trying fallback")

                st.session_state.chat_history.append({"role": "assistant", "content": bot_reply})
                st.chat_message("assistant").write(bot_reply)
            except Exception:
                try:
                    import google.generativeai as old_genai
                    old_genai.configure(api_key=api_key)
                    model = old_genai.GenerativeModel("gemini-1.5-flash")
                    response = model.generate_content(context)
                    bot_reply = response.text

                    st.session_state.chat_history.append({"role": "assistant", "content": bot_reply})
                    st.chat_message("assistant").write(bot_reply)
                except Exception as e:
                    error_msg = str(e)
                    if "429" in error_msg or "quota" in error_msg.lower() or "RESOURCE_EXHAUSTED" in error_msg:
                        st.error("🚫 **API Quota Exceeded**\n\nThe Gemini API free tier has daily/minute limits. Please try again in a few minutes or upgrade your API key plan.\n\n[Check your quota](https://ai.dev/rate-limit)")
                    elif "404" in error_msg:
                        st.error("⚙️ AI model temporarily unavailable. Please try again later.")
                    elif "api key" in error_msg.lower() or "invalid" in error_msg.lower():
                        st.error(f"🔑 **Invalid API Key**\n\nPlease check your API key in secrets.toml.")
                    else:
                        st.error(f"❌ **AI Error**: {error_msg[:200]}..." if len(error_msg) > 200 else f"❌ **AI Error**: {error_msg}")

else:
    # Initial state - show instructions
    st.info("👆 Configure your flight search in the sidebar and click **FIND FLIGHTS** to see available options with risk analysis.")

    # Show travel tips
    with st.expander("💡 How to use Flight Risk AI"):
        st.markdown("""
        1. **Select Origin and Destination** - Choose your departure and arrival airports
        2. **Set Travel Date** - Pick your preferred travel date
        3. **Add Travel Details** - Specify your travel purpose and any commitments
        4. **Click FIND FLIGHTS** - Get a list of flights with AI-powered risk analysis
        5. **Review Recommendations** - Our AI will suggest the best options based on your needs
        6. **Ask Questions** - Use the Travel Assistant for personalized advice
        """)
