import streamlit as st
import firebase_admin
from firebase_admin import credentials, firestore
from engine.humanizer import TurnitoutEngine
import json

# Hide the Streamlit menu and footer so it looks like it belongs on your website
st.set_page_config(page_title="TurnItOut AI", layout="centered")
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# 1. Initialize Firebase Securely
# 1. Initialize Firebase Securely
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

# 2. Get the secret token and user ID from the invisible URL
user_uid = st.query_params.get("uid")
secret_token = st.query_params.get("token")

if not user_uid or not secret_token:
    st.error("🔒 Access Denied. Please use this tool through the official TurnItOut dashboard.")
    st.stop()

# 3. Verify the token against Firebase
user_ref = db.collection("users").document(user_uid)
user_doc = user_ref.get()

if not user_doc.exists or user_doc.to_dict().get("secureToken") != secret_token:
    st.error("🔒 Invalid or Expired Session. Please log in again from the main dashboard.")
    st.stop()

# 4. Load User Quota Data
user_data = user_doc.to_dict()
plan = user_data.get("plan", "free")
daily_used = user_data.get("dailyHumanizeUsed", 0)
reward_credits = user_data.get("rewardCredits", 0)

# Apply Free Tier Limits
if plan == "free" and daily_used >= 3 and reward_credits <= 0:
    st.warning("⚠️ You have used all your free runs! Please upgrade to Pro on the main dashboard to continue.")
    st.stop()

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

    with st.spinner("AI is rewriting your text..."):
        # Run your NLP Engine
        output_text, telemetry = engine.humanize_text(
            text=input_text,
            intensity=intensity,
            register=tone,
            use_zipf_synsets=True,
            clean_em_dashes=True,
            voice_profile=None,
            adaptive_mode=True,
            seed=None # Let it do all 12 loops, Streamlit has the RAM for it!
        )
        
        # Deduct Quota in Firebase instantly
        if reward_credits > 0:
             user_ref.update({"rewardCredits": reward_credits - 1, "dailyWordsUsed": firestore.Increment(word_count)})
        else:
             user_ref.update({"dailyHumanizeUsed": daily_used + 1, "dailyWordsUsed": firestore.Increment(word_count)})
             
        # Display Result
        st.success("Text Successfully Humanized!")
        st.write(output_text)