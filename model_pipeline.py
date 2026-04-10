import pandas as pd
import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import os

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text) # Remove punctuation
    return text

def train_models():
    if not os.path.exists("data/youtube_dataset.csv"):
        print("Error: CSV file not found. Run scraper.py first with a valid API key.")
        return

    print("Loading data...")
    df = pd.read_csv("data/youtube_dataset.csv")
    
    if df.empty:
        print("Error: Dataset is empty. Check your scraper or API key.")
        return

    # 1. Preprocessing
    df['combined_text'] = df['title'] + " " + df['description']
    df['cleaned_text'] = df['combined_text'].apply(clean_text)
    
    # 2. NLP Feature Extraction
    print("Vectorizing text...")
    vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
    X_text = vectorizer.fit_transform(df['cleaned_text']).toarray()
    
    # 3. Unsupervised Learning: Clustering
    print("Clustering content...")
    num_clusters = min(5, len(df)) # Adjust clusters if dataset is very small
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_text)
    
    # 4. Supervised Learning: Predicting RPM Proxy
    print("Training predictive model...")
    y = df['value_score']
    X_train, X_test, y_train, y_test = train_test_split(X_text, y, test_size=0.2, random_state=42)
    
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    score = rf_model.score(X_test, y_test)
    print(f"Model R^2 Score: {score:.2f}")
    
    # 5. Save models for the app
    joblib.dump(vectorizer, 'data/vectorizer.pkl')
    joblib.dump(kmeans, 'data/kmeans.pkl')
    joblib.dump(rf_model, 'data/rf_model.pkl')
    print("Models saved successfully in /data folder.")

if __name__ == "__main__":
    train_models()