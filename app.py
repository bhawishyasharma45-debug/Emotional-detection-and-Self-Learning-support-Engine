import os
import json
import hashlib
import pandas as pd
from datetime import datetime
import streamlit as st
import google.generativeai as genai
from src.preprocessing import clean_text, EmotionPredictor, EMOTION_RESPONSES
from src.bert_model import BERTEmotionClassifier

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
USERS_FILE = "users.json"

# ─────────────────────────────────────────────
# USER PERSISTENCE HELPERS
# ─────────────────────────────────────────────
def _hash_password(password: str) -> str:
    """Return SHA-256 hex digest of the password."""
    return hashlib.sha256(password.encode()).hexdigest()


def load_users() -> dict:
    """Load users from JSON file. Seeds default accounts on first run."""
    if not os.path.exists(USERS_FILE):
        default_users = {
            "student1": {
                "password": _hash_password("password123"),
                "email": "student1@example.com",
                "created_at": datetime.now().isoformat(),
            },
            "naman_gaur": {
                "password": _hash_password("securepass"),
                "email": "naman@example.com",
                "created_at": datetime.now().isoformat(),
            },
        }
        save_users(default_users)
        return default_users
    with open(USERS_FILE, "r") as f:
        return json.load(f)


def save_users(users: dict) -> None:
    """Persist users dict to JSON file."""
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)


def register_user(username: str, email: str, password: str) -> tuple[bool, str]:
    """
    Register a new user. Returns (success, message).
    Validates uniqueness of username and email.
    """
    users = load_users()
    if username in users:
        return False, "Username already exists. Please choose another."
    if any(u["email"].lower() == email.lower() for u in users.values()):
        return False, "An account with this email already exists."
    users[username] = {
        "password": _hash_password(password),
        "email": email,
        "created_at": datetime.now().isoformat(),
    }
    save_users(users)
    return True, "Account created successfully!"


def authenticate_user(username: str, password: str) -> bool:
    """Return True if username/password pair is valid."""
    users = load_users()
    if username not in users:
        return False
    return users[username]["password"] == _hash_password(password)


# ─────────────────────────────────────────────
# SESSION STATE INITIALIZATION
# ─────────────────────────────────────────────
def _init_session():
    defaults = {
        "emotion_history": [],
        "authenticated": False,
        "username": "",
        "auth_page": "signin",   # "signin" | "signup"
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


_init_session()

# ─────────────────────────────────────────────
# GEMINI CONFIGURATION
# ─────────────────────────────────────────────
try:
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        try:
            api_key = st.secrets.get("GEMINI_API_KEY", "")
        except Exception:
            api_key = ""
    genai.configure(api_key=api_key)
    model_gemini = genai.GenerativeModel("gemini-2.5-flash")
except Exception:
    model_gemini = None

# ─────────────────────────────────────────────
# MODEL LOADING (CACHED)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    try:
        bilstm_model = EmotionPredictor("models/blstm/bilstm_student_adaptive.keras")
        bert_model = None
        try:
            bert_model = BERTEmotionClassifier()
            bert_model.load_model("models/bert_emotion_model_final")
        except Exception:
            pass
        return bilstm_model, bert_model, "✅ Models loaded"
    except Exception as e:
        return None, None, f"❌ Error: {e}"


# ─────────────────────────────────────────────
# UTILITY FUNCTIONS
# ─────────────────────────────────────────────
def get_gemini_response(field: str, problem: str, emotion: str, confidence: float) -> str:
    try:
        prompt = f"""You are a helpful learning assistant. A student studying {field} is feeling \
{emotion} (confidence: {confidence:.1%}) about this problem:

"{problem}"

Provide a clear, supportive response with:
1. Brief acknowledgment of their feeling
2. One specific tip or strategy for {field}
3. One encouraging next step

Use simple, clear language. Keep each point to 1-2 sentences. No markdown formatting."""
        response = model_gemini.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"AI response unavailable: {e}"


def save_to_csv(field: str, problem: str, emotion: str, confidence: float, ai_response: str) -> bool:
    """Append a new interaction to the CSV learning store."""
    try:
        new_example = {
            "text": problem,
            "emotion": emotion.lower(),
            "confidence": confidence,
            "response": ai_response,
            "field": field,
            "timestamp": datetime.now().isoformat(),
        }
        csv_path = "emotion_response_examples.csv"
        if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
            df = pd.read_csv(csv_path)
            df = pd.concat([df, pd.DataFrame([new_example])], ignore_index=True)
        else:
            df = pd.DataFrame([new_example])
        df.to_csv(csv_path, index=False)

        mapping_path = "emotion_response_mapping.csv"
        if os.path.exists(mapping_path) and os.path.getsize(mapping_path) > 0:
            mapping_df = pd.read_csv(mapping_path)
            if emotion not in mapping_df["emotion"].values:
                mapping_df = pd.concat(
                    [mapping_df, pd.DataFrame([{"emotion": emotion, "response": ai_response}])],
                    ignore_index=True,
                )
                mapping_df.to_csv(mapping_path, index=False)
        else:
            pd.DataFrame([{"emotion": emotion, "response": ai_response}]).to_csv(mapping_path, index=False)

        return True
    except Exception as e:
        st.error(f"Failed to save to CSV: {e}")
        return False


def get_mixed_emotions(scores: dict, threshold: float = 0.15) -> list:
    sorted_emotions = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    primary = sorted_emotions[0]
    mixed = [primary] + [(name, score) for name, score in sorted_emotions[1:] if score >= threshold]
    return mixed if len(mixed) > 1 else [primary]


def add_to_history(field, problem, emotion, confidence, ai_response, bilstm_scores, bert_result=None):
    bilstm_mixed = get_mixed_emotions(bilstm_scores)
    emotion_label = " + ".join(em[0] for em in bilstm_mixed) if len(bilstm_mixed) > 1 else emotion

    st.session_state.emotion_history.append({
        "timestamp": datetime.now(),
        "field": field,
        "problem": problem,
        "emotion": emotion_label,
        "confidence": confidence,
        "ai_response": ai_response,
        "all_scores": bilstm_scores,
        "model": "BiLSTM",
    })

    if bert_result:
        bert_mixed = get_mixed_emotions(bert_result["scores"])
        bert_label = " + ".join(em[0] for em in bert_mixed) if len(bert_mixed) > 1 else bert_result["emotion"]
        st.session_state.emotion_history.append({
            "timestamp": datetime.now(),
            "field": field,
            "problem": problem,
            "emotion": bert_label,
            "confidence": bert_result["confidence"],
            "ai_response": ai_response,
            "all_scores": bert_result["scores"],
            "model": "BERT",
        })


# ─────────────────────────────────────────────
# PAGE CONFIG (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Learning Support Engine",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Load models after page config
bilstm_model, bert_model, status_msg = load_models()


# ─────────────────────────────────────────────
# AUTH VIEWS
# ─────────────────────────────────────────────
def show_auth_page():
    """Renders the Sign In / Sign Up tabbed authentication page."""
    # Center the auth card
    _, center, _ = st.columns([1, 1.6, 1])
    with center:
        st.markdown(
            "<h1 style='text-align:center;margin-bottom:0.2rem;'>🎓 Learning Support Engine</h1>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='text-align:center;color:gray;margin-bottom:1.5rem;'>"
            "Emotion-aware AI tutoring for students</p>",
            unsafe_allow_html=True,
        )

        tab_signin, tab_signup = st.tabs(["🔑  Sign In", "📝  Sign Up"])

        # ── SIGN IN TAB ──
        with tab_signin:
            with st.form("signin_form"):
                username = st.text_input("Username", placeholder="Enter your username")
                password = st.text_input("Password", type="password", placeholder="Enter your password")
                submitted = st.form_submit_button("Sign In", use_container_width=True, type="primary")

            if submitted:
                if not username or not password:
                    st.error("Please fill in all fields.")
                elif authenticate_user(username, password):
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.success(f"Welcome back, **{username}**! 🎉")
                    st.rerun()
                else:
                    st.error("Invalid username or password. Please try again.")

        # ── SIGN UP TAB ──
        with tab_signup:
            with st.form("signup_form"):
                new_username  = st.text_input("Username", placeholder="Choose a username (min. 3 chars)", key="su_user")
                new_email     = st.text_input("Email", placeholder="your@email.com", key="su_email")
                new_password  = st.text_input("Password", type="password",
                                              placeholder="Min. 8 characters", key="su_pass")
                confirm_pass  = st.text_input("Confirm Password", type="password",
                                              placeholder="Repeat your password", key="su_confirm")
                register_btn  = st.form_submit_button("Create Account", use_container_width=True, type="primary")

            if register_btn:
                # ── Input validation ──
                errors = []
                if not new_username or not new_email or not new_password or not confirm_pass:
                    errors.append("All fields are required.")
                else:
                    if len(new_username) < 3:
                        errors.append("Username must be at least 3 characters.")
                    if "@" not in new_email or "." not in new_email:
                        errors.append("Please enter a valid email address.")
                    if len(new_password) < 8:
                        errors.append("Password must be at least 8 characters.")
                    if new_password != confirm_pass:
                        errors.append("Passwords do not match.")

                if errors:
                    for err in errors:
                        st.error(err)
                else:
                    success, message = register_user(new_username, new_email, new_password)
                    if success:
                        st.success(f"✅ {message} Please switch to Sign In to log in.")
                        st.balloons()
                    else:
                        st.error(message)


# ─────────────────────────────────────────────
# DASHBOARD VIEW
# ─────────────────────────────────────────────
def show_dashboard():
    st.title("🎓 Emotion & Learning Support Engine")
    st.write(f"Welcome back, **{st.session_state.username}**! Analyze your learning state instantly.")

    # ── Sidebar ──
    with st.sidebar:
        st.header("📊 Dashboard")
        st.write(f"Active User: `{st.session_state.username}`")
        st.write(f"Models: {status_msg}")
        st.write(f"Total Interactions: {len(st.session_state.emotion_history)}")

        csv_path = "emotion_response_examples.csv"
        csv_count = 0
        if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
            csv_count = len(pd.read_csv(csv_path))
        st.write(f"CSV Examples: {csv_count}")

        col_clear, col_logout = st.columns(2)
        with col_clear:
            if st.button("Clear History", use_container_width=True):
                st.session_state.emotion_history = []
                st.rerun()
        with col_logout:
            if st.button("Sign Out", type="secondary", use_container_width=True):
                st.session_state.authenticated = False
                st.session_state.username = ""
                st.rerun()

        st.subheader("Recent Sessions")
        recent = st.session_state.emotion_history[-3:]
        for item in reversed(recent):
            st.write(f"• {item['field']}: {item['emotion']} ({item['confidence']:.1%})")

    st.divider()

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📝 Student Input Portal")
        field = st.selectbox(
            "What field are you studying?",
            ["Computer Science", "Mathematics", "Physics", "Chemistry", "Biology",
             "Engineering", "Business", "Literature", "History", "Psychology", "Other"],
            help="Select your area of study for personalized responses",
        )
        problem = st.text_area(
            f"Describe your {field} problem or challenge:",
            placeholder=f"e.g., 'I'm struggling with algorithms in {field}'",
            height=120,
        )

    with col2:
        st.subheader("⚙️ Settings")
        use_ai      = st.checkbox("Use AI Response (Gemini)", value=True)
        save_data   = st.checkbox("Save to CSV for learning", value=True)
        show_details = st.checkbox("Show analysis details", value=False)

    st.write("")

    if st.button("🔍 Get AI Learning Help", type="primary", use_container_width=True):
        if problem.strip():
            with st.spinner("Analyzing your learning state..."):
                bilstm_result = bilstm_model.predict(problem)
                bert_result   = bert_model.predict(problem) if bert_model else None

                emotion    = bilstm_result["emotion"]
                confidence = bilstm_result["confidence"]

                if use_ai and model_gemini is not None:
                    ai_response = get_gemini_response(field, problem, emotion, confidence)
                else:
                    ai_response = EMOTION_RESPONSES[emotion]["response"]

                if save_data:
                    save_to_csv(field, problem, emotion, confidence, ai_response)

                add_to_history(field, problem, emotion, confidence, ai_response,
                               bilstm_result["scores"], bert_result)

                if show_details:
                    st.divider()
                    detail_col1, detail_col2 = st.columns([1, 1])

                    with detail_col1:
                        st.write("**BiLSTM Model Matrix**")
                        bilstm_mixed = get_mixed_emotions(bilstm_result["scores"])
                        if len(bilstm_mixed) > 1:
                            mixed_text = " + ".join(
                                f"{EMOTION_RESPONSES[em[0]]['emoji']} {em[0]}" for em in bilstm_mixed
                            )
                            st.metric("Mixed Emotions", mixed_text, f"Primary: {bilstm_mixed[0][1]:.1%}")
                        else:
                            emoji = EMOTION_RESPONSES[emotion]["emoji"]
                            st.metric("Emotion", f"{emoji} {emotion}", f"{confidence:.1%}")
                        for emo_name, score in sorted(bilstm_result["scores"].items(), key=lambda x: x[1], reverse=True):
                            st.progress(score, text=f"{emo_name}: {score:.1%}")

                    if bert_result:
                        with detail_col2:
                            st.write("**BERT Transformer**")
                            bert_mixed = get_mixed_emotions(bert_result["scores"])
                            if len(bert_mixed) > 1:
                                mixed_text = " + ".join(
                                    f"{EMOTION_RESPONSES[em[0]]['emoji']} {em[0]}" for em in bert_mixed
                                )
                                st.metric("Mixed Emotions", mixed_text, f"Primary: {bert_mixed[0][1]:.1%}")
                            else:
                                bert_emoji = EMOTION_RESPONSES[bert_result["emotion"]]["emoji"]
                                st.metric("Emotion", f"{bert_emoji} {bert_result['emotion']}",
                                          f"{bert_result['confidence']:.1%}")
                            for emo_name, score in sorted(bert_result["scores"].items(), key=lambda x: x[1], reverse=True):
                                st.progress(score, text=f"{emo_name}: {score:.1%}")

            st.divider()
            st.success("**Empathetic AI Response:**")
            st.write(ai_response)
        else:
            st.warning("Please enter a valid text statement to process.")


# ─────────────────────────────────────────────
# APPLICATION ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    if st.session_state.authenticated:
        show_dashboard()
    else:
        show_auth_page()