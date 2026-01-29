import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import plotly.graph_objects as go
import os
import json
from pathlib import Path
from datetime import datetime, time

# ============================================
# 1. CONFIGURATION & STYLES
# ============================================
st.set_page_config(page_title="Flight Risk AI", page_icon="✈️", layout="wide")

st.markdown("""
    <style>
    .risk-card {
        padding: 20px;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stButton>button { width: 100%; font-weight: bold; border-radius: 8px; height: 3em; }
    </style>
    """, unsafe_allow_html=True)

# ============================================
# 2. LOAD GEMINI API & DEBUG CALLER
# ============================================
try:
    api_key = st.secrets["GEMINI_API_KEY"]
except (FileNotFoundError, KeyError):
    try:
        import toml
        secrets_path = os.path.join(os.path.dirname(__file__), "secrets.toml")
        with open(secrets_path, 'r') as f:
            secrets = toml.load(f)
        api_key = secrets['GEMINI_API_KEY']
    except:
        api_key = None

def get_gemini_response_safe(context, api_key):
    """
    Uses 'gemini-1.5-flash' with high temperature for creative/smart roleplay.
    """
    # --- CHANGED: Added Config for Creativity ---
    generation_config = {
        "temperature": 0.8,  # Higher = More personality/creativity
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 1024,
    }

    try:
        # Try new google.genai client first
        from google import genai as new_genai

        try:
            client = new_genai.Client(api_key=api_key) if api_key else new_genai.Client()
        except Exception:
            try:
                new_genai.configure(api_key=api_key)
                client = new_genai
            except Exception:
                client = new_genai.Client()

        # Use client.models.generate_content with available model
        if hasattr(client, 'models') and hasattr(client.models, 'generate_content'):
            resp = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=context
            )
            return resp.text
        else:
            raise Exception("Client does not support models.generate_content")

    except Exception:
        # Fallback to older library
        try:
            import google.generativeai as old_genai
            old_genai.configure(api_key=api_key)
            model = old_genai.GenerativeModel(
                "gemini-1.5-flash",
                generation_config=generation_config
            )
            response = model.generate_content(context)
            return response.text
        except Exception as e:
            error_msg = str(e)

            if "429" in error_msg or "quota" in error_msg.lower() or "RESOURCE_EXHAUSTED" in error_msg:
                return "🚫 **API Quota Exceeded**\n\nThe Gemini API has daily/minute limits. Please try again in a few minutes or upgrade your API key.\n\n[Check quota](https://ai.dev/rate-limit)"
            elif "404" in error_msg:
                return "⚙️ Model temporarily unavailable. Please try again later."
            elif "api key" in error_msg.lower() or "invalid" in error_msg.lower():
                return "🔑 **Invalid API Key**\n\nPlease check your API key in secrets.toml."
            else:
                return f"❌ AI Error: {error_msg[:200]}..." if len(error_msg) > 200 else f"❌ AI Error: {error_msg}"

# ============================================
# 3. LOAD ARTIFACTS
# ============================================
@st.cache_resource
def load_artifacts():
    base_path = os.path.join(os.path.dirname(__file__), "flight_risk_app")

    model_file = os.path.join(base_path, "flight_risk_model.json")
    scaler_file = os.path.join(base_path, "flight_risk_scaler.pkl")
    enc_file = os.path.join(base_path, "flight_risk_encodings.pkl")
    traffic_file = os.path.join(base_path, "flight_traffic_stats.pkl")
    dist_file = os.path.join(base_path, "flight_distance_lookup.pkl")

    try:
        if not os.path.exists(model_file):
            st.error(f"❌ Missing file: {model_file}")
            return None, None, None, None, None

        model = xgb.Booster()
        model.load_model(model_file)

        scaler = joblib.load(scaler_file)
        encodings = joblib.load(enc_file)
        traffic_stats = joblib.load(traffic_file)
        dist_lookup = joblib.load(dist_file)

        return model, scaler, encodings, traffic_stats, dist_lookup

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return None, None, None, None, None

model, scaler, encodings, traffic_stats, dist_lookup = load_artifacts()

if not model:
    st.stop()

# ============================================
# 4. EXPANDED PERSONA LOGIC (10 PROFILES)
# ============================================
PERSONA_SOLUTIONS = {
    "The Executive": {
        "user_label": "🏢 Executive / Business Pro",
        "voice_sample": "I have a board meeting at 2 PM. I don't care about the cost, just get me there.",
        "tipping_point": "Delay > 30 mins (Schedule disruption)",
        "coping_style": "Productivity Protection. Find the nearest Lounge. Get on Wi-Fi immediately.",
        "flight_strategy": "Double-book. Buy a refundable seat on a competitor leaving 1 hour later.",
        "insurance_archetype": {
            "name": "Corporate Shield Elite",
            "focus": "Time Assurance",
            "payout": "100% Instant Refund + $1,000 Incidental Coverage.",
            "fine_print": "Auto-payout triggered by 30 min delay.",
            "price_multiplier": 5.0
        },
        "tone": "Direct, authoritative, efficiency-focused."
    },
    "The Student": {
        "user_label": "🎒 Student / Young Traveler",
        "voice_sample": "I have like $20 left for food. Can I sleep in the terminal?",
        "tipping_point": "Delay > 12 hours (Requires hotel)",
        "coping_style": "Survival Mode. Locate sleeping spots. Fill water bottles. Download movies.",
        "flight_strategy": "Sit tight. Never cancel unless the airline pays you to do it.",
        "insurance_archetype": {
            "name": "Backpacker Delay-Pay",
            "focus": "Food & Hostel Cover",
            "payout": "$50 instant cash for meals/hostels.",
            "fine_print": "$50 deductible. Receipts required.",
            "price_multiplier": 0.4
        },
        "tone": "Casual, frugal, resourceful. Use slang."
    },
    "The Parent": {
        "user_label": "🧸 Family / Parent",
        "voice_sample": "I have two toddlers and we are running out of diapers. I cannot do a 4-hour layover.",
        "tipping_point": "Delay > 60 mins (Kids getting restless)",
        "coping_style": "Chaos Containment. Find the 'Family Bathroom' and empty gates to run kids around.",
        "flight_strategy": "Book a day-rate hotel room immediately if delay > 2 hours. Don't power through.",
        "insurance_archetype": {
            "name": "Family Harmony Flex",
            "focus": "Sanity Protection",
            "payout": "Reimburses airport hotel, diapers, and DoorDash.",
            "fine_print": "Covers whole family. No receipts for food under $50.",
            "price_multiplier": 1.5
        },
        "tone": "Empathetic, practical, reassuring. 'We got this'."
    },
    "The Retiree": {
        "user_label": "👴 Retiree / Senior Traveler",
        "voice_sample": "My knees can't handle sitting on these hard chairs for three hours.",
        "tipping_point": "Delay > 2 hours (Physical discomfort)",
        "coping_style": "Comfort First. Request wheelchair assistance for priority seating. Find a quiet corner.",
        "flight_strategy": "Call the travel agent. Avoid rebooking on apps. Prioritize comfort over speed.",
        "insurance_archetype": {
            "name": "Silver Comfort Guard",
            "focus": "Health & Comfort",
            "payout": "Lounge access included + Medical evacuation if needed.",
            "fine_print": "24/7 Phone Concierge support included.",
            "price_multiplier": 2.5
        },
        "tone": "Respectful, patient, clear. No jargon."
    },
    "The Tourist": {
        "user_label": "📸 Tourist / Vacationer",
        "voice_sample": "We're going to miss our dinner reservation in Rome! This is supposed to be fun.",
        "tipping_point": "Delay > 4 hours (Losing vacation time)",
        "coping_style": "Experience Salvage. Research airport attractions or nearby city excursions.",
        "flight_strategy": "Check if the airline will re-route you through a different fun city.",
        "insurance_archetype": {
            "name": "Vacation Saver",
            "focus": "Experience Protection",
            "payout": "Reimburses missed tours and pre-paid hotels.",
            "fine_print": "Covers non-refundable bookings.",
            "price_multiplier": 1.0
        },
        "tone": "Upbeat, adventurous, helpful."
    },
    "The Digital Nomad": {
        "user_label": "💻 Digital Nomad / Remote Worker",
        "voice_sample": "I need stable upload speed. I have a Zoom call with a client in 45 minutes.",
        "tipping_point": "Delay > 0 mins (If Wi-Fi is bad)",
        "coping_style": "Tech Hunt. Find the gate with the best power outlets and 5G signal.",
        "flight_strategy": "Don't board if you can't work. Rebook for a plane with verified Wi-Fi.",
        "insurance_archetype": {
            "name": "Remote Work Shield",
            "focus": "Tech & Connection",
            "payout": "Pays for airport lounge Wi-Fi and lost billable hours.",
            "fine_print": "Requires proof of freelance work.",
            "price_multiplier": 1.2
        },
        "tone": "Tech-savvy, efficient, focused on 'Connectivity'."
    },
    "The Explorer": {
        "user_label": "🧗 Adventure / Explorer",
        "voice_sample": "It's all part of the journey. I can sleep on the floor if I have to.",
        "tipping_point": "Delay > 24 hours (Missing the expedition)",
        "coping_style": "Go with the Flow. Talk to locals. Read a book. Don't stress.",
        "flight_strategy": "Look for weird alternate routes (trains/buses).",
        "insurance_archetype": {
            "name": "Explorer Hazard Pay",
            "focus": "Gear & Activity",
            "payout": "Covers lost hiking gear and missed connection transport.",
            "fine_print": "High deductible. Geared for remote travel.",
            "price_multiplier": 0.8
        },
        "tone": "Chill, stoic, adventurous."
    },
    "The VIP": {
        "user_label": "💎 VIP / Luxury Traveler",
        "voice_sample": "This is unacceptable. Get me the concierge line. I don't wait in lines.",
        "tipping_point": "Delay > 15 mins (Inconvenience)",
        "coping_style": "Total Outsourcing. Call the Amex Centurion concierge. Have them fix it.",
        "flight_strategy": "Private Transfer. If commercial fails, look for a jet charter or first-class swap.",
        "insurance_archetype": {
            "name": "Black Card Infinite",
            "focus": "Exclusivity",
            "payout": "Private transfer to destination. No limits.",
            "fine_print": "Invitation only. $5000 premium.",
            "price_multiplier": 10.0
        },
        "tone": "Polite but demanding, sophisticated, 'White Glove'."
    },
    "The Immigrant": {
        "user_label": "🌍 Immigrant / Expat",
        "voice_sample": "I have 3 checked bags with gifts for my family. I cannot lose them.",
        "tipping_point": "Delay > 2 hours (Risk of baggage loss)",
        "coping_style": "Asset Protection. Stand near the gate desk to ensure luggage is tracked.",
        "flight_strategy": "Do not change airlines unless absolutely necessary (baggage nightmare).",
        "insurance_archetype": {
            "name": "Global Roots Cover",
            "focus": "Baggage & Visa",
            "payout": "$2000 per lost bag. Visa expiration protection.",
            "fine_print": "High baggage limits. Covers documentation fees.",
            "price_multiplier": 1.3
        },
        "tone": "Serious, protective, detail-oriented."
    },
    "The Commuter": {
        "user_label": "🚆 Commuter / Short-Haul",
        "voice_sample": "I just want to get home for dinner. I do this flight every Tuesday.",
        "tipping_point": "Delay > 90 mins (Misses dinner)",
        "coping_style": "Routine Optimization. You know the airport shortcuts. Go to your usual spot.",
        "flight_strategy": "Rent a car. If it's short-haul, driving is often faster than a 2-hour delay.",
        "insurance_archetype": {
            "name": "Commuter Express",
            "focus": "On-Time Guarantee",
            "payout": "Free rental car if flight delayed > 2 hours.",
            "fine_print": "Valid only for flights under 500 miles.",
            "price_multiplier": 1.1
        },
        "tone": "Brisk, experienced, 'Done this a million times'."
    }
}

# ============================================
# 5. OPPORTUNITY COST MATH ENGINE
# ============================================
def calculate_opportunity_cost(persona_key, delay_minutes):
    """
    Calculates the 'Pain Value' (in $) based on the unique mathematical profile of each user.
    """
    hours = delay_minutes / 60

    # 1. HIGH VALUE LINEAR (Time = Money)
    if persona_key == "The Executive":
        return (hours * 500) + (200 if hours > 2 else 0)

    elif persona_key == "The VIP":
        return (hours * 2000) # Extremely high hourly value

    elif persona_key == "The Commuter":
        # Moderate value, but spikes at "Dinner Time" (3 hours)
        cost = hours * 100
        if hours > 3: cost += 300
        return cost

    # 2. LOW FINANCIAL / HIGH TOLERANCE
    elif persona_key == "The Student":
        return (hours * 15) + (25 if hours > 4 else 0) # Food cost

    elif persona_key == "The Explorer":
        return (hours * 10) # Very low cost, part of the adventure

    # 3. EXPONENTIAL STRESS (Sanity = Money)
    elif persona_key == "The Parent":
        # Starts slow, explodes after 1.5 hours
        return 50 * (1.8 ** hours)

    elif persona_key == "The Retiree":
        # Physical pain factor increases over time
        return 40 * (1.5 ** hours)

    # 4. SITUATIONAL / SPECIFIC
    elif persona_key == "The Digital Nomad":
        # Cost of lost billable hours + Panic if meeting missed
        cost = hours * 80
        if hours > 1: cost += 150 # Missed Meeting
        return cost

    elif persona_key == "The Tourist":
        # Cost of vacation day lost (e.g. $400/day)
        return hours * (400 / 12) # Hourly portion of vacation cost

    elif persona_key == "The Immigrant":
        # Binary Risk: Low initially, but Baggage Loss/Visa risk is huge
        if hours < 4: return hours * 20
        else: return 1500 # Risk of baggage separation / chaos

    return hours * 50 # Default fallback

# ============================================
# 6. DYNAMIC PROMPT & QUOTE GENERATORS
# ============================================
def generate_insurance_quote(risk_prob, persona_key):
    """Generates a synthetic insurance price based on XGBoost risk + Persona Wealth."""
    base_price = 25.0
    risk_multiplier = 1 + (risk_prob * 1.5)
    wealth_multiplier = PERSONA_SOLUTIONS[persona_key]['insurance_archetype']['price_multiplier']
    final_price = base_price * risk_multiplier * wealth_multiplier
    return round(final_price, 2)

def get_solution_prompt(persona_key, risk_data, quote_price, user_question, opp_cost):
    """
    NEW STRUCTURE: Forces Roleplay and Intelligence.
    """
    p_data = PERSONA_SOLUTIONS[persona_key]
    ins = p_data['insurance_archetype']

    return f"""
    ### SYSTEM INSTRUCTION: ROLEPLAY MODE
    You are NOT a generic AI. You are a specific character described below.
    You MUST adopt the voice, slang, and priorities of this persona.

    ### 1. YOUR PERSONA: {persona_key}
    - **Voice/Style:** {p_data['tone']}
    - **Example Sentence:** "{p_data['voice_sample']}"
    - **What Stresses You:** {p_data['tipping_point']} (If the flight delay is close to this, act worried).

    ### 2. REAL-TIME DATA (Use this to be smart)
    - **Flight Risk Level:** {risk_data['class']} ({risk_data['prob']:.1%})
    - **Main Delay Reason:** {risk_data['reason']}
    - **Financial Pain:** The user is losing ~${opp_cost:.0f} worth of time/value right now.

    ### 3. THE USER'S QUESTION
    "{user_question}"

    ### 4. YOUR REQUIRED RESPONSE FORMAT
    Do not use standard AI intros like "Here is an analysis." Jump straight into character.

    **Section 1: The Strategic Move**
    First, look at the Flight Risk Level and Financial Pain:

    - If Flight Risk is LOW (probability < 25%) OR Financial Pain < $100:
      - Say clearly that there is no need for aggressive moves.
      - Recommend staying on the current flight and maybe light back-up checks only (e.g. monitor status, know alternative options).
    - If Flight Risk is MODERATE (25–60%) AND Financial Pain is between $100 and $500:
      - Suggest cautious optimizations (earlier arrival to airport, checking same-airline alternatives), but avoid expensive changes.
    - If Flight Risk is HIGH (>= 60%) OR Financial Pain >= $500:
      - You may recommend your default strategy: "{p_data['flight_strategy']}" (e.g., double-booking) and explain why it’s justified.
    - If the persona is "The Executive" and the delay probability is at least 40%:
      - Treat this as high enough risk to justify recommending "{p_data['flight_strategy']}" as a strong option, while still acknowledging cost trade-offs.

    Always reference the ${opp_cost:.0f} cost explicitly to justify whether a move is worth it or not.

    **Section 2: Survival Tactics**
    Give tips based on this vibe: "{p_data['coping_style']}".
    *Make the tips specific to a '{risk_data['reason']}' delay.*

    **Section 3: The Insurance Upsell**
    Pitch this exact product (do not make up a random one):
    * **Product:** {ins['name']} for ${quote_price}
    * **Pitch:** {ins['focus']}
    * **The Catch:** Mention "{ins['fine_print']}" so they trust you.

    **Constraint:** Keep it under 200 words. Be sharp. Stay in character.
    """

def get_chat_prompt(persona_key, risk_data, user_question, opp_cost, chat_history):
    """
    Conversational mode: Responds naturally to user questions while staying in character.
    """
    p_data = PERSONA_SOLUTIONS[persona_key]

    # Build conversation context from history
    history_context = ""
    if chat_history:
        history_context = "\n### PREVIOUS CONVERSATION:\n"
        for msg in chat_history[-6:]:  # Last 3 exchanges
            role = "User" if msg["role"] == "user" else "You"
            history_context += f"{role}: {msg['content']}\n"

    return f"""
    ### SYSTEM INSTRUCTION: CONVERSATIONAL TRAVEL ASSISTANT
    You are a travel assistant AI with a specific personality. Answer naturally and helpfully.
    DO NOT use the structured format (Strategic Move, Survival Tactics, etc.) - just chat naturally.

    ### YOUR PERSONA: {persona_key}
    - **Voice/Style:** {p_data['tone']}
    - **Character Example:** "{p_data['voice_sample']}"
    - **Key Concern:** {p_data['tipping_point']}

    ### CONTEXT (Use to give smart, relevant answers)
    - **Current Flight Risk:** {risk_data['class']} ({risk_data['prob']:.1%})
    - **Main Delay Reason:** {risk_data['reason']}
    - **User's Time Cost:** ~${opp_cost:.0f}
    {history_context}

    ### USER'S QUESTION
    "{user_question}"

    ### INSTRUCTIONS
    - Answer the user's question directly and conversationally
    - Stay in character ({persona_key} personality)
    - Be helpful, specific, and relevant to their flight situation
    - If they ask about insurance, you can mention {p_data['insurance_archetype']['name']}
    - Keep response under 150 words. Be natural, not robotic.
    """

# ============================================
# 7. PREDICTION ENGINE (XGBoost)
# ============================================
def get_encoded_value(col_name, category, encodings_dict):
    mapping = encodings_dict.get(col_name, {})
    default = encodings_dict.get(f'{col_name}_default', 0.20)
    return mapping.get(category, default)

def calibrate_prob(raw_prob):
    if raw_prob < 0.50: return raw_prob * 0.6
    else: return raw_prob

def predict_flight_risk(origin, dest, date_obj, time_obj):
    sched_hour = time_obj.hour
    day_of_week = date_obj.weekday()
    month = date_obj.month

    hour_sin = np.sin(2 * np.pi * sched_hour / 24)
    hour_cos = np.cos(2 * np.pi * sched_hour / 24)
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    day_sin = np.sin(2 * np.pi * day_of_week / 7)

    is_weekend = 1 if day_of_week >= 5 else 0
    is_holiday = 1 if month in [12, 1, 7] else 0

    is_morning = 1 if 5 <= sched_hour <= 11 else 0
    origin_dest = f"{origin}_{dest}"
    origin_morning = f"{origin}_Morning_{is_morning}"
    route_morning = f"{origin_dest}_{is_morning}"

    avg_route = traffic_stats['route_means'].get(origin_dest, traffic_stats['global_mean'])
    avg_origin = traffic_stats['origin_means'].get(origin, traffic_stats['global_mean'])

    sim_ewm_route = avg_route
    sim_ewm_origin = avg_origin
    sim_ewm_dest = avg_route

    route_delta = 0.0
    origin_delta = 0.0

    # dist_lookup uses lowercase keys
    distance = dist_lookup.get((origin.lower(), dest.lower()), dist_lookup['global_default'])

    origin_enc = get_encoded_value('origin', origin, encodings)
    od_enc = get_encoded_value('origin_dest', origin_dest, encodings)
    om_enc = get_encoded_value('origin_morning', origin_morning, encodings)
    rm_enc = get_encoded_value('route_morning', route_morning, encodings)

    data = {
        'distance': distance,
        'ewm_route_delay': sim_ewm_route,
        'ewm_origin_delay': sim_ewm_origin,
        'ewm_dest_delay': sim_ewm_dest,
        'route_traffic_delta': route_delta,
        'origin_traffic_delta': origin_delta,
        'hour_sin': hour_sin, 'hour_cos': hour_cos,
        'month_sin': month_sin, 'month_cos': month_cos,
        'day_sin': day_sin,
        'is_holiday': is_holiday, 'is_weekend': is_weekend,
        'origin_encoded': origin_enc,
        'origin_dest_encoded': od_enc,
        'origin_morning_encoded': om_enc,
        'route_morning_encoded': rm_enc
    }

    df_single = pd.DataFrame([data])
    num_cols = [
        'distance', 'ewm_route_delay', 'ewm_origin_delay', 'ewm_dest_delay',
        'route_traffic_delta', 'origin_traffic_delta'
    ]
    df_single[num_cols] = scaler.transform(df_single[num_cols])

    dmatrix = xgb.DMatrix(df_single)
    raw_prob = model.predict(dmatrix)[0]
    display_prob = calibrate_prob(raw_prob)

    return raw_prob, display_prob, is_morning, avg_route

# ============================================
# 8. UI & MAIN LOGIC
# ============================================
st.sidebar.header("✈️ Flight Search")

# --- PERSONA SELECTOR ---
st.sidebar.subheader("👤 Traveler Profile")
label_to_key = {v['user_label']: k for k, v in PERSONA_SOLUTIONS.items()}
selected_label = st.sidebar.selectbox("Select your travel mode:", list(label_to_key.keys()))
selected_persona_key = label_to_key[selected_label]
# ------------------------

st.sidebar.divider()

# Load airports from JSON file
@st.cache_data
def load_airports_data():
    """Load airport data from airports.json."""
    airports_path = Path(__file__).parent / "data" / "airports.json"
    try:
        with open(airports_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        st.warning("⚠️ airports.json not found. Using fallback data.")
        return []

airports_data = load_airports_data()

# Build lookup dictionaries from airports.json
AIRPORT_INFO = {a['airport_code']: a for a in airports_data}  # code -> full info
AIRPORT_CITIES = {a['airport_code']: a['city_name'] for a in airports_data}  # code -> city name

# Build city -> airports mapping from airports.json
city_to_airports = {}
for airport in airports_data:
    city = airport['city_name']
    code = airport['airport_code']
    if city not in city_to_airports:
        city_to_airports[city] = []
    city_to_airports[city].append(code)

# Get route keys for filtering
route_keys = [k for k in dist_lookup.keys() if k != 'global_default']
# Convert lowercase keys to uppercase for matching
all_airports_in_routes = set([k[0].upper() for k in route_keys] + [k[1].upper() for k in route_keys])

# Filter cities to only those with airports in our routes data
city_to_airports_filtered = {}
for city, codes in city_to_airports.items():
    valid_codes = [c for c in codes if c.upper() in all_airports_in_routes]
    if valid_codes:
        city_to_airports_filtered[city] = valid_codes

# Get unique cities sorted (only cities with airports in routes)
unique_cities = sorted(city_to_airports_filtered.keys())

# All airports available (for fallback options)
all_airports = sorted(list(all_airports_in_routes))

# Helper function to format airport display
def format_airport_option(code):
    """Format airport code with name for display."""
    info = AIRPORT_INFO.get(code)
    if info:
        return f"{code} - {info['airport_name']}"
    return code

# City selection
st.sidebar.subheader("📍 Route Selection")
selected_city = st.sidebar.selectbox(
    "Select City",
    options=[""] + unique_cities,
    index=0,
    help="Select a city to filter available airports"
)

# Origin selection - enabled only when city is selected
if selected_city:
    # Show ONLY airports from the selected city
    available_origins = sorted(city_to_airports_filtered.get(selected_city, []))

    if available_origins:
        # Format options with airport names
        origin_options_display = [format_airport_option(a) for a in available_origins]

        selected_origin_display = st.sidebar.selectbox(
            "Origin Airport",
            options=origin_options_display,
            index=0,
            help=f"Airports in {selected_city}"
        )

        # Extract airport code from display string
        origin = selected_origin_display.split(" - ")[0] if selected_origin_display else None
    else:
        st.sidebar.selectbox(
            "Origin Airport",
            options=["No airports found in this city"],
            disabled=True
        )
        origin = None
else:
    st.sidebar.selectbox(
        "Origin Airport",
        options=["Select a city first"],
        disabled=True
    )
    origin = None

# Destination selection - enabled only when origin is selected
if origin:
    # Get all possible destinations from this origin (convert to uppercase for display)
    possible_dests = sorted(list(set([k[1].upper() for k in route_keys if k[0].upper() == origin.upper()])))

    if possible_dests:
        dest_options_display = [format_airport_option(d) for d in possible_dests]
        selected_dest_display = st.sidebar.selectbox(
            "Destination Airport",
            options=dest_options_display,
            index=0,
            help=f"{len(possible_dests)} destinations available from {origin}"
        )
        dest = selected_dest_display.split(" - ")[0] if selected_dest_display else None
    else:
        st.sidebar.selectbox(
            "Destination Airport",
            options=["No routes available from this airport"],
            disabled=True
        )
        dest = None
else:
    st.sidebar.selectbox(
        "Destination Airport",
        options=["Select origin first"],
        disabled=True
    )
    dest = None

travel_date = st.sidebar.date_input("Date", value=datetime.today())
default_time = time(9, 0)
flight_time = st.sidebar.time_input("Time", value=default_time)

st.title("🛫 Flight Risk AI")
if origin and dest:
    st.caption(f"Predictive Analysis: {origin} ➝ {dest}")
else:
    st.caption("Select a city and airports to analyze flight risk")

if "result" not in st.session_state:
    st.session_state.result = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "opp_cost" not in st.session_state:
    st.session_state.opp_cost = 0

# Only show analyze button when both origin and dest are selected
if origin and dest:
    if st.button("🔍 ANALYZE FLIGHT RISK", type="primary"):
        with st.spinner("🔍 Analyzing flight risk... Please wait..."):
            # Step 1: Analyze schedule patterns
            raw_prob, display_prob, is_morning, avg_delay_mins = predict_flight_risk(origin, dest, travel_date, flight_time)

            # Qualitative traffic profile for explanation (today vs. typical)
            if avg_delay_mins <= 10:
                traffic_profile = "well below typical congestion for this route."
            elif avg_delay_mins <= 25:
                traffic_profile = "around normal congestion levels for this route."
            else:
                traffic_profile = "historically one of the more delay-prone routes."

            # --- REFINED DECISION LOGIC (WYSIWYG) ---
            if display_prob >= 0.60:
                risk_class = "HIGH"
                color = "#dc3545"  # Red
                reason = f"High probability of delay due to route congestion history. Historically, this route is {traffic_profile}"
            elif display_prob >= 0.30:
                risk_class = "MODERATE"
                color = "#ffc107"  # Yellow/Orange
                reason = f"Elevated risk. Minor delays (15-30 mins) are possible. Historically, this route is {traffic_profile}"
            else:
                risk_class = "LOW"
                color = "#28a745"  # Green
                reason = f"Normal schedule risk. On-time arrival likely. Historically, this route is {traffic_profile}"

            # Morning 'cold start' note: only soften messaging, do NOT upgrade risk class
            morning_cold_start_note = None
            if is_morning and display_prob >= 0.25:
                morning_cold_start_note = (
                    "🌅 Morning Flight: traffic is clear, but early departures sometimes face 'cold start' mechanical checks. "
                    "Overall risk is still relatively low for this route."
                )
            elif is_morning:
                morning_cold_start_note = (
                    "🌅 Morning Flight: historically quite reliable at this time. Cold start risk is minimal today."
                )

            st.session_state.result = {
                "prob": display_prob,
                "raw_prob": raw_prob,
                "class": risk_class,
                "color": color,
                "reason": reason,
                "is_morning": is_morning,
                "avg_delay": avg_delay_mins,
                "morning_note": morning_cold_start_note,
            }

            # Calculate & Store Opportunity Cost
            st.session_state.opp_cost = calculate_opportunity_cost(selected_persona_key, avg_delay_mins)
            st.session_state.chat_history = []

            # Step 2: Auto-generate initial recommendations
            if api_key:
                insurance_price = generate_insurance_quote(display_prob, selected_persona_key)
                initial_prompt = get_solution_prompt(
                    persona_key=selected_persona_key,
                    risk_data=st.session_state.result,
                    quote_price=insurance_price,
                    user_question="What should I do about this flight?",
                    opp_cost=st.session_state.opp_cost
                )
                initial_response = get_gemini_response_safe(initial_prompt, api_key)
                st.session_state.chat_history.append({"role": "assistant", "content": initial_response})
                st.session_state.initial_reco_shown = True
else:
    # Show instructions when origin/dest not fully selected
    st.info("👆 Select a **City**, then **Origin Airport**, and finally **Destination Airport** in the sidebar to analyze flight risk.")

# --- RESULTS DISPLAY ---
if st.session_state.result:
    res = st.session_state.result

    # 1. Risk Card
    st.markdown(f"""
    <div class="risk-card" style="background-color: {res['color']};">
        <h1 style="margin:0; font-size: 3em;">{res['class']} RISK</h1>
        <h3 style="margin:0;">Probability: {res['prob']:.1%}</h3>
        <hr style="border-top: 1px solid rgba(255,255,255,0.3);">
        <p style="font-size: 1.2em;">{res['reason']}</p>
    </div>
    """, unsafe_allow_html=True)

    # 2. Details
    c1, c2 = st.columns([1, 1])
    with c1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=res['prob'] * 100,
            title={'text': "Delay Probability (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "white"},
                'steps': [
                    {'range': [0, 30], 'color': "#28a745"},
                    {'range': [30, 60], 'color': "#ffc107"},
                    {'range': [60, 100], 'color': "#dc3545"}
                ]
            }
        ))
        fig.update_layout(height=250, margin=dict(t=50, b=0, l=20, r=20))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("📊 Route Stats")
        st.write(f"**Typical Delay:** {res['avg_delay']:.0f} mins (Historical Avg)")
        if res.get("morning_note"):
            st.info(res["morning_note"])
        elif res['is_morning']:
            st.info("🌅 Morning Flight: 'Cold Start' risk logic applied.")
        else:
            st.success("☀️ Day/Evening Flight: Standard traffic logic applied.")

    # 3. AI Assistant
    st.divider()
    st.subheader(f"💬 {selected_persona_key} Assistant")

    for msg in st.session_state.chat_history:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("Ask about connections, insurance, or backup options..."):
        if not api_key:
            st.error("⚠️ Gemini API Key missing.")
        else:
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)

            # Use conversational mode for user questions
            opp_cost_val = st.session_state.opp_cost

            # Generate conversational prompt (not structured format)
            chat_prompt = get_chat_prompt(
                persona_key=selected_persona_key,
                risk_data=res,
                user_question=prompt,
                opp_cost=opp_cost_val,
                chat_history=st.session_state.chat_history
            )

            with st.spinner(f"{selected_persona_key} is thinking..."):
                reply = get_gemini_response_safe(chat_prompt, api_key)

            st.session_state.chat_history.append({"role": "assistant", "content": reply})
            st.chat_message("assistant").write(reply)
