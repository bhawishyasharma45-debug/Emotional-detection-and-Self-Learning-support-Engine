import os
import pandas as pd
from datetime import datetime
import streamlit as st
import google.generativeai as genai
from src.preprocessing import clean_text, EmotionPredictor
from src.bert_model import BERTEmotionClassifier

# Emoji dictionary mapping for mixed sentiment visualization
EMOTION_RESPONSES = {
    'Bored': {'emoji': '🥱'},
    'Confident': {'emoji': '😎'},
    'Confused': {'emoji': '😕'},
    'Curious': {'emoji': '🤔'},
    'Frustrated': {'emoji': '😤'}
}

# Configure Gemini Generative Model Backend tracking the system requirements
try:
    genai.configure(api_key=os.getenv("GEMINI_API_KEY", ""))
    model_gemini = genai.GenerativeModel("gemini-1.5-flash")
except Exception:
    # Graceful fallback initialization layer for localized development execution
    class MockGemini:
        def generate_content(self, prompt):
            class MockResponse:
                @property
                def text(self):
                    return "Acknowledgment: I see you're working through this concept. Tip: Break the core architecture into separate functional files. Next step: Try testing each small component individually."
            return MockResponse()
    model_gemini = MockGemini()

@st.cache_resource
def load_models():
    try:
        bilstm_model = EmotionPredictor("models/blstm/bilstm_student_adaptive.keras")
        
        # Load BERT model if available
        bert_model = None
        try:
            bert_model = BERTEmotionClassifier()
            bert_model.load_model('models/bert_emotion_model_final')
        except:
            pass
            
        return bilstm_model, bert_model, "✅ Models loaded"
    except Exception as e:
        return None, None, f"❌ Error: {e}"

# --- VERBATIM FROM 6c0e41f8-aabd-4bdb-b0a3-c81533462c79 ---
def get_gemini_response(field, problem, emotion, confidence):
    """Get AI response from Gemini based on field and problem."""
    try:
        prompt = f"""
You are a helpful learning assistant. A student studying {field} is feeling {emotion} (confidence: {confidence:.1%}) about this problem:

"{problem}"

Provide a clear, supportive response with:
1. Brief acknowledgment of their feeling
2. One specific tip or strategy for {field}
3. One encouraging next step

Use simple, clear language. Keep each point to 1-2 sentences. No markdown formatting.
"""
        response = model_gemini.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"AI response unavailable: {e}"

def save_to_csv(field, problem, emotion, confidence, ai_response):
    """Save new interaction to CSV files handling empty files safely."""
    try:
        new_example = {
            'text': problem,
            'emotion': emotion.lower(),
            'confidence': confidence,
            'response': ai_response,
            'field': field,
            'timestamp': datetime.now().isoformat()
        }
        
        if os.path.exists("emotion_response_examples.csv") and os.path.getsize("emotion_response_examples.csv") > 0:
            df = pd.read_csv("emotion_response_examples.csv")
            df = pd.concat([df, pd.DataFrame([new_example])], ignore_index=True)
        else:
            df = pd.DataFrame([new_example])
            
        df.to_csv("emotion_response_examples.csv", index=False)
        return True
    except:
        return False

# Function to detect mixed sentiment matching explicit blueprints
def get_mixed_emotions(scores, threshold=0.15):
    sorted_emotions = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    primary = sorted_emotions[0]
    mixed = [primary]
    
    for emotion, score in sorted_emotions[1:]:
        if score >= threshold:
            mixed.append((emotion, score))
            
    return mixed if len(mixed) > 1 else [primary]

# Initialize engines
bilstm_model, bert_model, status_msg = load_models()

# UI Header Configuration
st.set_page_config(page_title="Learning Support Engine", page_icon="🎓", layout="wide")
st.title("🎓 Emotion & Learning Support Engine")
st.write("Analyze student sentiment states and generate empathetic AI feedback responses instantly.")
st.sidebar.info(status_msg)

st.divider()

def main():
    # Core Layout Columns
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📝 Student Input Portal")
        
        # --- VERBATIM FROM 8972760b-0763-4a3e-be44-49755a4845a7 ---
        # Field selection
        field = st.selectbox(
            "What field are you studying?",
            ["Computer Science", "Mathematics", "Physics", "Chemistry", "Biology", 
             "Engineering", "Business", "Literature", "History", "Psychology", "Other"],
            help="Select your area of study for personalized responses"
        )
        
        # Problem description
        problem = st.text_area(
            f"Describe your {field} problem or challenge:",
            placeholder=f"e.g., 'I'm struggling with algorithms in {field}' or 'This concept is confusing'",
            height=120
        )
        
    st.write("") # Layout spacer spacing matching form baselines

    # --- VERBATIM FROM 91b68041-3a1d-4d82-a9e1-90d14f10faab ---
    # Analysis button
    if st.button("🔍 Get AI Learning Help", type="primary", use_container_width=True):
        if problem.strip():
            with st.spinner("Analyzing your learning state..."):
                # Get predictions from models
                bilstm_result = bilstm_model.predict(problem)
                bert_result = bert_model.predict(problem) if bert_model else None
                
                # Use BiLSTM result for AI response (or you can choose based on confidence)
                emotion_result = bilstm_result
                emotion = emotion_result['emotion']
                confidence = emotion_result['confidence']
                
                # Trigger live Gemini generation prompt block mapping
                ai_feedback = get_gemini_response(field, problem, emotion, confidence)
                
                # --- BiLSTM Schema Render Block ---
                with col1:
                    st.write("**BiLSTM Model Matrix**")
                    bilstm_mixed = get_mixed_emotions(bilstm_result['scores'])
                    
                    if len(bilstm_mixed) > 1:
                        mixed_text = " + ".join([f"{EMOTION_RESPONSES[em[0]]['emoji']} {em[0]}" for em in bilstm_mixed])
                        st.metric("Mixed Emotions", mixed_text, f"Primary: {bilstm_mixed[0][1]:.1%}")
                    else:
                        bilstm_emoji = EMOTION_RESPONSES[bilstm_result['emotion']]['emoji']
                        st.metric("Emotion", f"{bilstm_emoji} {bilstm_result['emotion']}", f"{bilstm_result['confidence']:.1%}")
                        
                    for emotion_name, score in sorted(bilstm_result['scores'].items(), key=lambda x: x[1], reverse=True):
                        st.progress(score, text=f"{emotion_name}: {score:.1%}")
                
                # --- BERT Schema Render Block ---
                if bert_result:
                    with col2:
                        st.write("**BERT Transformer**")
                        bert_mixed = get_mixed_emotions(bert_result['scores'])
                        
                        if len(bert_mixed) > 1:
                            mixed_text = " + ".join([f"{EMOTION_RESPONSES[em[0]]['emoji']} {em[0]}" for em in bert_mixed])
                            st.metric("Mixed Emotions", mixed_text, f"Primary: {bert_mixed[0][1]:.1%}")
                        else:
                            bert_emoji = EMOTION_RESPONSES[bert_result['emotion']]['emoji']
                            st.metric("Emotion", f"{bert_emoji} {bert_result['emotion']}", f"{bert_result['confidence']:.1%}")
                            
                        for emotion_name, score in sorted(bert_result['scores'].items(), key=lambda x: x[1], reverse=True):
                            st.progress(score, text=f"{emotion_name}: {score:.1%}")
                
                # Sync logs to persist file data trackers
                save_to_csv(field, problem, emotion, confidence, ai_feedback)
                
                # Unified Response Presentation
                st.divider()
                st.success("**Empathetic AI Response:**")
                st.write(ai_feedback)
        else:
            st.warning("Please enter a valid text statement to process.")

if __name__ == "__main__":
    main()