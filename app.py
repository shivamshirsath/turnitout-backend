import streamlit as st
import firebase_admin
from firebase_admin import credentials, auth, firestore

# 1. Initialize Firebase Admin securely (Using your Streamlit secrets)
if not firebase_admin._apps:
    cert_dict = dict(st.secrets["firebase"])
    cred = credentials.Certificate(cert_dict)
    firebase_admin.initialize_app(cred)

db = firestore.client()

# 2. Extract the JWT token from the Next.js Iframe URL
query_params = st.query_params
id_token = query_params.get("token", None)

if not id_token:
    st.error("🚨 Unauthorized Access. Please log in through the main dashboard.")
    st.stop()

# 3. Cryptographically Verify Token (0 Database Reads!)
try:
    decoded_token = auth.verify_id_token(id_token)
    uid = decoded_token['uid']
except Exception as e:
    st.error("🚨 Invalid or expired session. Please refresh the dashboard.")
    st.stop()

# 4. RAM Caching: Only read from database ONCE per session
if "user_data" not in st.session_state:
    user_ref = db.collection('users').document(uid)
    doc = user_ref.get()
    if doc.exists:
        st.session_state.user_data = doc.to_dict()
    else:
        st.error("User profile not found in database.")
        st.stop()

# 5. Display Credits strictly in Streamlit UI
user_plan = st.session_state.user_data.get('plan', 'free')
daily_used = st.session_state.user_data.get('dailyHumanizeUsed', 0)
reward_credits = st.session_state.user_data.get('rewardCredits', 0)

# Define plan limits
limits = {"free": 3, "pro": 50, "ultra": 300}
max_runs = limits.get(user_plan, 3)
runs_left = max_runs - daily_used + reward_credits

st.sidebar.success(f"🔋 **Runs Left:** {runs_left}")
st.sidebar.info(f"💎 **Plan:** {user_plan.upper()}")

# 6. The Humanize Button Logic (1 Read, 1 Write)
text_input = st.text_area("Paste text to humanize:")
if st.button("Humanize Text"):
    if runs_left <= 0:
        st.error("❌ You have run out of credits! Please upgrade your plan.")
    else:
        with st.spinner("Humanizing..."):
            # -> INSERT YOUR ENCRYPTED AI PIPELINE CODE HERE <-
            # humanized_result = engine.process(text_input)
            
            # Update Database (1 Write)
            user_ref = db.collection('users').document(uid)
            
            if daily_used < max_runs:
                # Consume normal daily limit
                user_ref.update({'dailyHumanizeUsed': firestore.Increment(1)})
                st.session_state.user_data['dailyHumanizeUsed'] += 1
            else:
                # Consume reward credit
                user_ref.update({'rewardCredits': firestore.Increment(-1)})
                st.session_state.user_data['rewardCredits'] -= 1

            st.success("Text Humanized Successfully!")
            # st.write(humanized_result)
            
            # Force UI to update the credit counter immediately
            st.rerun()