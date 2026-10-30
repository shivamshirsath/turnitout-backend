import streamlit as st
import firebase_admin
from firebase_admin import credentials, auth as firebase_auth, firestore
from engine.humanizer import TurnitoutEngine

# Hide the Streamlit menu and footer so it looks like it belongs on your website
st.set_page_config(page_title="TurnItOut AI", layout="centered")
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# 1. Initialize Firebase Securely (Using your exact working secrets)
if not firebase_admin._apps:
    cred = credentials.Certificate({
        "type": "service_account",
        "project_id": st.secrets["FIREBASE_PROJECT_ID"],
        "private_key": st.secrets["FIREBASE_PRIVATE_KEY"].replace('\\n', '\n'),
        "client_email": st.secrets["FIREBASE_CLIENT_EMAIL"],
        "token_uri": "https://oauth2.googleapis.com/token",
    })
    firebase_admin.initialize_app(cred)

db = firestore.client()

# 2. Extract the Cryptographic JWT Token (0 Database Reads!)
id_token = st.query_params.get("token")

if not id_token:
    st.error("🔒 Access Denied. Please use this tool through the official TurnItOut dashboard.")
    st.stop()

# 3. Cryptographically Verify Token 
try:
    decoded_token = firebase_auth.verify_id_token(id_token)
    user_uid = decoded_token['uid']
except Exception as e:
    st.error("🔒 Invalid or Expired Session. Please refresh the main dashboard.")
    st.stop()

# --- THE CACHE BUSTER ---
# If Next.js sends a new timestamp, or the user clicks Sync, drop the old RAM cache
sync_param = st.query_params.get("sync", "0")
if st.session_state.get("last_sync") != sync_param:
    st.session_state.last_sync = sync_param
    st.session_state.pop("user_data", None) # Force a fresh read!

# 4. RAM Caching: Read from Firestore ONLY ONCE per session
if "user_data" not in st.session_state:
    user_ref = db.collection("users").document(user_uid)
    user_doc = user_ref.get()
    if user_doc.exists:
        st.session_state.user_data = user_doc.to_dict()
    else:
        st.error("User profile not found in database.")
        st.stop()

# Extract cached user data
plan = st.session_state.user_data.get("plan", "free")
daily_used = st.session_state.user_data.get("dailyHumanizeUsed", 0)
reward_credits = st.session_state.user_data.get("rewardCredits", 0)

# Calculate limits
limits = {"free": 3, "pro": 50, "ultra": 300}
max_runs = limits.get(plan, 3)
runs_left = max_runs - daily_used + reward_credits

# Display Credits and Sync Button directly on the main screen
col_plan, col_runs, col_sync = st.columns(3)

with col_plan:
    st.info(f"💎 **Plan:** {plan.upper()}")
with col_runs:
    st.success(f"🔋 **Runs Left:** {runs_left}")
with col_sync:
    if st.button("🔄 Sync Account", use_container_width=True):
        st.session_state.pop("user_data", None)
        st.rerun()

# 5. Load Engine (Cached so it only loads once)
@st.cache_resource
def load_engine():
    return TurnitoutEngine()

engine = load_engine()

# --- THE USER INTERFACE ---
input_text = st.text_area("Paste the text you want to humanize here:", height=200)

col1, col2 = st.columns(2)
with col1:
    intensity = st.slider("Humanize Intensity", 1, 100, 95)
with col2:
    tone = st.selectbox("Tone", ["professional", "academic", "casual"])

if st.button("Humanize Text", use_container_width=True):
    if not input_text.strip():
        st.warning("Please enter some text.")
        st.stop()
        
    word_count = len(input_text.split())
    if plan == "free" and word_count > 350:
        st.error(f"Word limit exceeded! Free plan allows 350 words. You entered {word_count}.")
        st.stop()

    if runs_left <= 0:
        st.error("❌ You have used all your runs! Please upgrade to Pro on the main dashboard to continue.")
        st.stop()

    with st.spinner("AI is rewriting your text..."):
        # Run NLP Engine
        output_text, telemetry = engine.humanize_text(
            text=input_text,
            intensity=intensity,
            register=tone,
            use_zipf_synsets=True,
            clean_em_dashes=True,
            voice_profile=None,
            adaptive_mode=True,
            seed=None 
        )
        
        # Deduct Quota in Firebase (1 Write)
        user_ref = db.collection("users").document(user_uid)
        
        if daily_used < max_runs:
            user_ref.update({
                "dailyHumanizeUsed": firestore.Increment(1), 
                "dailyWordsUsed": firestore.Increment(word_count)
            })
            st.session_state.user_data["dailyHumanizeUsed"] += 1
        else:
            user_ref.update({
                "rewardCredits": firestore.Increment(-1), 
                "dailyWordsUsed": firestore.Increment(word_count)
            })
            st.session_state.user_data["rewardCredits"] -= 1
            
        # 🚨 THE FIX: Save the output to RAM before rebooting the UI
        st.session_state.humanized_result = output_text
        
        # Force UI counter to update instantly
        st.rerun()

# 🚨 THE FIX: Display the result safely outside the button click!
if "humanized_result" in st.session_state:
    st.success("Text Successfully Humanized!")
    st.write(st.session_state.humanized_result)