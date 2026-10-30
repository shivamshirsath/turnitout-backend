import io
import os
import zipfile
import streamlit as st
import firebase_admin
from firebase_admin import credentials, auth as firebase_auth, firestore
from engine.humanizer import TurnitoutEngine
from processors.doc_processor import process_docx_in_place
from processors.latex_processor import process_latex_manuscript

# Set page title to humantext
st.set_page_config(page_title="humantext workspace", layout="centered")

# --- MASSIVE UI THEME OVERRIDE ---
st.markdown("""
    <style>
    /* 1. Hide default Streamlit headers, footers, and menus completely */
    #MainMenu {visibility: hidden !important;}
    header {visibility: hidden !important;}
    footer {visibility: hidden !important;}
    .stApp bottom {display: none !important;}

    /* 2. Base Background matching Next.js Walnut Brown */
    .stApp {
        background-color: #1C140F !important;
    }
    
    /* 3. General Text & Typography */
    html, body, [class*="css"], p, span, h1, h2, h3, h4, h5, h6, label {
        font-family: 'Inter', sans-serif !important;
        color: #F1F5F9 !important;
    }

    /* 4. Text Area (Source Input) */
    .stTextArea textarea {
        background-color: #2A1E15 !important;
        border: 1px solid #5A3E2B !important;
        color: #F1F5F9 !important;
        border-radius: 12px !important;
        padding: 1rem !important;
    }
    .stTextArea textarea:focus {
        border-color: #D5851F !important;
        box-shadow: 0 0 0 1px #D5851F !important;
    }

    /* 5. Dropdowns and Sliders */
    .stSelectbox div[data-baseweb="select"] > div {
        background-color: #2A1E15 !important;
        border: 1px solid #5A3E2B !important;
        border-radius: 10px !important;
        color: #F1F5F9 !important;
    }
    .stSlider [data-testid="stTickBar"] {
        background-color: #5A3E2B !important;
    }
    .stSlider [data-baseweb="slider"] div {
        background-color: #87A96B !important;
    }

    /* 6. Primary Action Buttons (Amber Glow) */
    .stButton > button {
        background-color: #D5851F !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        padding: 0.5rem 1rem !important;
        transition: all 0.3s ease !important;
    }
    .stButton > button:hover {
        background-color: #b56e18 !important;
        box-shadow: 0 4px 15px rgba(213, 133, 31, 0.4) !important;
        transform: translateY(-1px) !important;
        color: white !important;
    }

    /* 7. Tabs Styling (Text vs File Upload) */
    .stTabs [data-baseweb="tab-list"] {
        background-color: transparent !important;
        gap: 2rem !important;
        border-bottom: 1px solid #5A3E2B !important;
    }
    .stTabs [data-baseweb="tab"] {
        color: #94A3B8 !important;
        background-color: transparent !important;
        border: none !important;
        padding-bottom: 1rem !important;
    }
    .stTabs [aria-selected="true"] {
        color: #87A96B !important;
        border-bottom: 2px solid #87A96B !important;
        font-weight: 700 !important;
    }

    /* 8. Alert Boxes (Info, Warning, Error, Success) */
    .stAlert {
        background-color: #2A1E15 !important;
        border: 1px solid #5A3E2B !important;
        border-radius: 12px !important;
        color: #F1F5F9 !important;
    }
    .stAlert p {
        color: #F1F5F9 !important;
    }

    /* 9. File Uploader Dropzone */
    [data-testid="stFileUploadDropzone"] {
        background-color: #2A1E15 !important;
        border: 2px dashed #5A3E2B !important;
        border-radius: 16px !important;
        padding: 2rem !important;
        transition: all 0.3s ease !important;
    }
    [data-testid="stFileUploadDropzone"]:hover {
        border-color: #87A96B !important;
        background-color: rgba(135, 169, 107, 0.05) !important;
    }
    [data-testid="stFileUploadDropzone"] button {
        background-color: #5A3E2B !important;
        color: white !important;
        border-radius: 8px !important;
    }
    </style>
""", unsafe_allow_html=True)

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

# 2. Extract and Verify Token
id_token = st.query_params.get("token")
if not id_token:
    st.error("🔒 Access Denied. Please use this tool through the official humantext dashboard.")
    st.stop()

try:
    decoded_token = firebase_auth.verify_id_token(id_token)
    user_uid = decoded_token['uid']
except Exception as e:
    st.error("🔒 Invalid or Expired Session. Please refresh the main dashboard.")
    st.stop()

# 3. Cache Buster
sync_param = st.query_params.get("sync", "0")
if st.session_state.get("last_sync") != sync_param:
    st.session_state.last_sync = sync_param
    st.session_state.pop("user_data", None)

# 4. Fetch User Data
if "user_data" not in st.session_state:
    user_ref = db.collection("users").document(user_uid)
    user_doc = user_ref.get()
    if user_doc.exists:
        st.session_state.user_data = user_doc.to_dict()
    else:
        st.error("User profile not found in database.")
        st.stop()

# --- NEW 6-TIER LIMITS DICTIONARY ---
PLAN_LIMITS = {
    "free":    {"runs": 2,   "words": 100,  "uploads": 1},
    "pass_3":  {"runs": 30,  "words": 1500, "uploads": 5},
    "pass_7":  {"runs": 50,  "words": 2000, "uploads": 10},
    "pass_14": {"runs": 75,  "words": 2500, "uploads": 15},
    "pro":     {"runs": 100, "words": 3000, "uploads": 30},
    "ultra":   {"runs": 200, "words": 5000, "uploads": 100}
}

plan = st.session_state.user_data.get("plan", "free")
daily_used = st.session_state.user_data.get("dailyHumanizeUsed", 0)
uploads_used = st.session_state.user_data.get("dailyUploadsUsed", 0)
reward_credits = st.session_state.user_data.get("rewardCredits", 0)

# Calculate active limits
max_runs = PLAN_LIMITS.get(plan, PLAN_LIMITS["free"])["runs"]
max_words = PLAN_LIMITS.get(plan, PLAN_LIMITS["free"])["words"]
max_uploads = PLAN_LIMITS.get(plan, PLAN_LIMITS["free"])["uploads"]

runs_left = (max_runs - daily_used) + reward_credits
uploads_left = max_uploads - uploads_used

# UI Header
col_plan, col_runs, col_sync = st.columns(3)
with col_plan:
    st.info(f"💎 **Plan:** {plan.upper()}")
with col_runs:
    st.success(f"🔋 **Runs Left:** {runs_left}")
with col_sync:
    if st.button("🔄 Sync Account", use_container_width=True):
        st.session_state.pop("user_data", None)
        st.rerun()

@st.cache_resource
def load_engine():
    return TurnitoutEngine()

engine = load_engine()
tab_text, tab_files = st.tabs(["📝 Text Humanizer", "📁 File Upload & Processing"])

# --- TEXT HUMANIZER TAB ---
with tab_text:
    input_text = st.text_area("Paste the text you want to humanize here:", height=200)

    col1, col2 = st.columns(2)
    with col1:
        intensity = st.slider("Humanize Intensity", 1, 100, 100)
    with col2:
        tone = st.selectbox("Tone", ["professional"])

    if st.button("Humanize Text", use_container_width=True):
        if not input_text.strip():
            st.warning("Please enter some text.")
            st.stop()
            
        word_count = len(input_text.split())
        if word_count > max_words:
            st.error(f"❌ Word limit exceeded! Your '{plan}' plan allows {max_words} words per request. You entered {word_count}.")
            st.stop()

        if runs_left <= 0:
            st.error("❌ You have used all your runs! Please upgrade your plan on the main dashboard to continue.")
            st.stop()

        with st.spinner("AI is rewriting your text..."):
            output_text, telemetry = engine.humanize_text(
                text=input_text, intensity=intensity, register=tone,
                use_zipf_synsets=True, clean_em_dashes=True,
                voice_profile=None, adaptive_mode=True, seed=None 
            )
            
            user_ref = db.collection("users").document(user_uid)
            if daily_used < max_runs:
                user_ref.update({"dailyHumanizeUsed": firestore.Increment(1), "dailyWordsUsed": firestore.Increment(word_count)})
                st.session_state.user_data["dailyHumanizeUsed"] += 1
            else:
                user_ref.update({"rewardCredits": firestore.Increment(-1), "dailyWordsUsed": firestore.Increment(word_count)})
                st.session_state.user_data["rewardCredits"] -= 1
                
            st.session_state.humanized_result = output_text
            st.rerun()

    if "humanized_result" in st.session_state:
        st.success("Text Successfully Humanized!")
        st.write(st.session_state.humanized_result)

# --- FILE UPLOAD TAB ---
with tab_files:
    if max_uploads == 0:
        st.error("⚠️ File uploads are not supported on the Free Starter plan. Please upgrade to a Pass or Pro subscription.")
    else:
        st.info(f"📁 You have **{uploads_left}** file uploads remaining today.")
        uploaded_files = st.file_uploader(
            "Upload .docx, .tex, .md, .txt, or .zip",
            type=["docx", "tex", "md", "txt", "zip"],
            accept_multiple_files=True
        )
        
        if uploaded_files and st.button("⚡ Process Uploaded Files", type="primary", use_container_width=True):
            cost = len(uploaded_files)
            
            if uploads_left < cost:
                st.error(f"❌ Not enough file upload credits! You are trying to upload {cost} files, but only have {uploads_left} left today.")
                st.stop()
            if runs_left < cost:
                st.error(f"❌ Not enough Humanizer Runs! Processing {cost} files requires {cost} runs.")
                st.stop()
                
            with st.spinner(f"Processing {cost} file(s) and securely updating quota..."):
                proc_intensity = 95
                proc_register = "professional"
                proc_zipf = True
                proc_dashes = True
                proc_adaptive = True
                proc_voice = None

                if len(uploaded_files) == 1 and not uploaded_files[0].name.endswith(".zip"):
                    uf = uploaded_files[0]
                    raw_b = uf.read()
                    if uf.name.endswith(".docx"):
                        out_b, tel = process_docx_in_place(raw_b, engine, proc_intensity, proc_register, proc_zipf, proc_voice)
                        st.session_state.dl_data = out_b
                        st.session_state.dl_mime = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                    elif uf.name.endswith(".tex"):
                        tex_str = raw_b.decode("utf-8", errors="ignore")
                        out_tex, tel = process_latex_manuscript(tex_str, engine, proc_intensity, proc_register, proc_zipf, proc_voice)
                        st.session_state.dl_data = out_tex.encode("utf-8")
                        st.session_state.dl_mime = "text/plain"
                    else:
                        txt_str = raw_b.decode("utf-8", errors="ignore")
                        out_txt, tel = engine.humanize_text(txt_str, proc_intensity, proc_register, proc_zipf, proc_dashes, proc_voice, adaptive_mode=proc_adaptive)
                        st.session_state.dl_data = out_txt.encode("utf-8")
                        st.session_state.dl_mime = "text/plain"
                    st.session_state.dl_name = f"humanized_{uf.name}"
                    
                else:
                    master_zip = io.BytesIO()
                    with zipfile.ZipFile(master_zip, "w", zipfile.ZIP_DEFLATED) as zout:
                        for uf in uploaded_files:
                            raw_b = uf.read()
                            if uf.name.endswith(".docx"):
                                h_b, _ = process_docx_in_place(raw_b, engine, proc_intensity, proc_register, proc_zipf, proc_voice)
                                zout.writestr(f"humanized_{uf.name}", h_b)
                            elif uf.name.endswith(".tex"):
                                h_tex, _ = process_latex_manuscript(raw_b.decode("utf-8", errors="ignore"), engine, proc_intensity, proc_register, proc_zipf, proc_voice)
                                zout.writestr(f"humanized_{uf.name}", h_tex.encode("utf-8"))
                            elif uf.name.endswith(".zip"):
                                with zipfile.ZipFile(io.BytesIO(raw_b), "r") as zin:
                                    for member in zin.namelist():
                                        if member.endswith("/") or "__MACOSX" in member: continue
                                        m_bytes = zin.read(member)
                                        base_name = os.path.basename(member)
                                        if member.endswith(".docx"):
                                            hb, _ = process_docx_in_place(m_bytes, engine, proc_intensity, proc_register, proc_zipf, proc_voice)
                                            zout.writestr(f"humanized_{base_name}", hb)
                                        elif member.endswith(".tex"):
                                            ht, _ = process_latex_manuscript(m_bytes.decode("utf-8", errors="ignore"), engine, proc_intensity, proc_register, proc_zipf, proc_voice)
                                            zout.writestr(f"humanized_{base_name}", ht.encode("utf-8"))
                                        elif member.endswith((".md", ".txt")):
                                            htxt, _ = engine.humanize_text(m_bytes.decode("utf-8", errors="ignore"), proc_intensity, proc_register, proc_zipf, proc_dashes, proc_voice, adaptive_mode=proc_adaptive)
                                            zout.writestr(f"humanized_{base_name}", htxt.encode("utf-8"))
                            else:
                                h_t, _ = engine.humanize_text(raw_b.decode("utf-8", errors="ignore"), proc_intensity, proc_register, proc_zipf, proc_dashes, proc_voice, adaptive_mode=proc_adaptive)
                                zout.writestr(f"humanized_{uf.name}", h_t.encode("utf-8"))
                                
                    st.session_state.dl_data = master_zip.getvalue()
                    st.session_state.dl_name = "humantext_bundle.zip"
                    st.session_state.dl_mime = "application/zip"

                # Deduct Quotas
                user_ref = db.collection("users").document(user_uid)
                user_ref.update({"dailyUploadsUsed": firestore.Increment(cost)})
                st.session_state.user_data["dailyUploadsUsed"] = uploads_used + cost

                for _ in range(cost):
                    if st.session_state.user_data.get("dailyHumanizeUsed", 0) < max_runs:
                        user_ref.update({"dailyHumanizeUsed": firestore.Increment(1)})
                        st.session_state.user_data["dailyHumanizeUsed"] += 1
                    else:
                        user_ref.update({"rewardCredits": firestore.Increment(-1)})
                        st.session_state.user_data["rewardCredits"] -= 1

                st.rerun()

        if "dl_data" in st.session_state:
            st.success("✅ Files successfully processed and quota deducted!")
            st.download_button(
                label=f"⬇️ Download {st.session_state.dl_name}",
                data=st.session_state.dl_data,
                file_name=st.session_state.dl_name,
                mime=st.session_state.dl_mime
            )