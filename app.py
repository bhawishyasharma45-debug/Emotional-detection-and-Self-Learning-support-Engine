import os
import pandas as pd
from datetime import datetime
import streamlit as st
from src.predict import EmotionOrchestrator
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

# --- VERBATIM FROM e142eb6e-cf04-4c75-90bf-c9bf65043d72 ---
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

# --- VERBATIM FROM e77f6b8b-e8c4-416f-8b66-44cd7e80c718 ---
def save_to_csv(field, problem, emotion, confidence, ai_response):
    """Save new interaction to CSV files."""
    try:
        # Update examples CSV
        new_example = {
            'text': problem,
            'emotion': emotion.lower(),
            'confidence': confidence,
            'response': ai_response,
            'field': field,
            'timestamp': datetime.now().isoformat()
        }
        
        # Check if file exists
        if os.path.exists("emotion_response_examples.csv"):
            df = pd.read_csv("emotion_response_examples.csv")
            df = pd.concat([df, pd.DataFrame([new_example])], ignore_index=True)
        else:
            df = pd.DataFrame([new_example])
            
        df.to_csv("emotion_response_examples.csv", index=False)
        
        # Update mapping CSV if new emotion response pair
        if os.path.exists("emotion_response_mapping.csv"):
            mapping_df = pd.read_csv("emotion_response_mapping.csv")
            if emotion not in mapping_df['emotion'].values:
                new_mapping = pd.DataFrame([{'emotion': emotion, 'response': ai_response}])
                mapping_df = pd.concat([mapping_df, new_mapping], ignore_index=True)
                mapping_df.to_csv("emotion_response_mapping.csv", index=False)
                
        return True
    except Exception as e:
        st.error(f"Failed to save to CSV: {e}")
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
orchestrator = EmotionOrchestrator()

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
        
        # Dropdown selection to feed the 'field' requirement for CSV logs
        academic_field = st.selectbox("Academic Department Field:", ["Computer Science", "Data Science", "Mathematics", "General"])
        
        student_text = st.text_area(
            "Enter the student's problem statement or comment below:",
            placeholder="Type here (e.g., I am completely stuck on this assignment and nothing makes sense...)",
            height=150
        )
        
        submit_btn = st.button("Analyze Sentiment State", type="primary")

    if submit_btn and student_text.strip():
        if bilstm_model:
            with st.spinner("Processing pipeline segments..."):
                # Clean text through pipeline
                cleaned_input = clean_text(student_text)
                
                # Run predictions through cached memory structures
                bilstm_result = bilstm_model.predict(cleaned_input)
                
                if bert_model:
                    bert_result = bert_model.predict(cleaned_input)
                else:
                    bert_result = None
                    
                orchestrator_results = orchestrator.analyze_student_state(student_text)
                
                # --- BiLSTM Schema Render ---
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
                
                # --- BERT Schema Render ---
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
                
                # --- PERSIST DATA TO CSV FILES ---
                save_to_csv(
                    field=academic_field,
                    problem=student_text,
                    emotion=orchestrator_results["dominant_emotion"],
                    confidence=bilstm_result["confidence"],
                    ai_response=orchestrator_results["ai_feedback"]
                )
                
                # Unified Response Presentation
                st.divider()
                st.success("**Empathetic AI Response:**")
                st.write(orchestrator_results["ai_feedback"])
        else:
            st.error("Backend pipeline engines are offline.")
    elif submit_btn:
        st.warning("Please enter a valid text statement to process.")

if __name__ == "__main__":
    main()