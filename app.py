import streamlit as st
from src.predict import EmotionOrchestrator

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
                
                # Display dominant state metrics
                st.metric(label="Detected Dominant Emotion", value=results["dominant_emotion"])
                
                # Show ensemble confidence chart
                st.write("**Ensemble Probability Metrics:**")
                st.bar_chart(results["predictions"])
                
                # Print Generative Response
                st.success("**Empathetic AI Response:**")
                st.write(results["ai_feedback"])
        else:
            st.error("Backend pipeline engine is offline.")
    elif submit_btn:
        st.warning("Please enter a valid text statement to process.")