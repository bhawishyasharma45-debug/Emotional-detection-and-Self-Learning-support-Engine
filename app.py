import streamlit as st
from src.predict import EmotionOrchestrator

# Emoji dictionary mapping for mixed sentiment visualization
EMOTION_RESPONSES = {
    'Bored': {'emoji': '🥱'},
    'Confident': {'emoji': '😎'},
    'Confused': {'emoji': '😕'},
    'Curious': {'emoji': '🤔'},
    'Frustrated': {'emoji': '😤'}
}

# Function to detect mixed sentiment matching Epic 3 workspace thresholds
def get_mixed_emotions(scores, threshold=0.15):
    sorted_emotions = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    primary = sorted_emotions[0]
    mixed = [primary]
    
    for emotion, score in sorted_emotions[1:]:
        if score >= threshold:
            mixed.append((emotion, score))
            
    return mixed if len(mixed) > 1 else [primary]

# Initialize the master engine backend
@st.cache_resource
def load_engine():
    return EmotionOrchestrator()

try:
    engine = load_engine()
except Exception as e:
    st.error(f"Engine Initialization Error: {e}")
    engine = None

# UI Header Configuration
st.set_page_config(page_title="Learning Support Engine", page_icon="🎓", layout="wide")
st.title("🎓 Emotion & Learning Support Engine")
st.write("Analyze student sentiment states and generate empathetic AI feedback responses instantly.")

st.divider()

# Core Layout Columns
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📝 Student Input Portal")
    student_text = st.text_area(
        "Enter the student's problem statement or comment below:",
        placeholder="Type here (e.g., I am completely stuck on this assignment and nothing makes sense...)",
        height=150
    )
    
    submit_btn = st.button("Analyze Sentiment State", type="primary")

with col2:
    st.subheader("📊 Diagnostic Outputs")
    
    if submit_btn and student_text.strip():
        if engine:
            with st.spinner("Processing pipeline segments..."):
                # Execute full backend pipeline
                results = engine.analyze_student_state(student_text)
                
                # Process scores for mixed emotion validation
                mixed_emotions = get_mixed_emotions(results["predictions"])
                
                # Dynamic Metric Rendering Block
                if len(mixed_emotions) > 1:
                    mixed_text = " + ".join([f"{EMOTION_RESPONSES[em[0]]['emoji']} {em[0]}" for em in mixed_emotions])
                    st.metric("Mixed Emotions", mixed_text, f"Primary: {mixed_emotions[0][1]:.1%}")
                else:
                    emo_name = mixed_emotions[0][0]
                    emo_score = mixed_emotions[0][1]
                    emoji = EMOTION_RESPONSES[emo_name]['emoji']
                    st.metric("Emotion", f"{emoji} {emo_name}", f"{emo_score:.1%}")
                
                # Show explicit breakdown via progress bars sorted by confidence score descending
                st.write("**Ensemble Probability Breakdown:**")
                for emotion_name, score in sorted(results["predictions"].items(), key=lambda x: x[1], reverse=True):
                    st.progress(score, text=f"{emotion_name}: {score:.1%}")
                
                st.divider()
                
                # Print Generative Response
                st.success("**Empathetic AI Response:**")
                st.write(results["ai_feedback"])
        else:
            st.error("Backend pipeline engine is offline.")
    elif submit_btn:
        st.warning("Please enter a valid text statement to process.")