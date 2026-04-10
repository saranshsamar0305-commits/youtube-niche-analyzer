import streamlit as st
import joblib
import pandas as pd
import re
st.set_page_config(page_title="Niche Analyzer", page_icon="📊", layout="wide")
# Load pre-trained models
@st.cache_resource
def load_models():
    try:
        vectorizer = joblib.load('data/vectorizer.pkl')
        kmeans = joblib.load('data/kmeans.pkl')
        rf_model = joblib.load('data/rf_model.pkl')
        return vectorizer, kmeans, rf_model
    except FileNotFoundError:
        return None, None, None

vectorizer, kmeans, rf_model = load_models()

def clean_text(text):
    text = str(text).lower()
    return re.sub(r'[^a-z\s]', '', text)

# Dashboard UI
st.title("📈 High-RPM Content Niche Analyzer")
st.markdown("Enter a potential YouTube video title and description to predict its content value score and niche cluster.")

if not vectorizer:
    st.warning("⚠️ Models not found. Please run `scraper.py` and then `model_pipeline.py` first.")
else:
    # User Inputs
    st.sidebar.header("Input Content Data")
    video_title = st.sidebar.text_input("Video Title", "Top 10 AI Hardware Trends")
    video_desc = st.sidebar.text_area("Video Description", "In this video, we explore the latest GPUs, NPUs, and custom silicon.")

    if st.sidebar.button("Analyze Content"):
        # Process input
        combined_text = clean_text(video_title + " " + video_desc)
        vectorized_text = vectorizer.transform([combined_text]).toarray()
        
        # Predictions
        cluster_id = kmeans.predict(vectorized_text)[0]
        predicted_score = rf_model.predict(vectorized_text)[0]
        
        # Display Results
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(label="Predicted Value Score", value=f"{predicted_score:.2f} / 100")
            st.progress(max(0.0, min(1.0, predicted_score / 100)))
            
        with col2:
            st.metric(label="Assigned Niche Cluster", value=f"Cluster {cluster_id}")
            
        st.subheader("Actionable Insight")
        if predicted_score > 50:
            st.success("High potential value! The algorithm detects strong engagement keywords.")
        else:
            st.warning("Lower potential value. Consider using stronger keywords.")