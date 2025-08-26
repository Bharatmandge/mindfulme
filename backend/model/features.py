import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from pathlib import Path

def build_tfidf():
    processed_path = Path(r"C:\Users\bhara\emotion-diary-ai\data\processed\combined_data_clean.csv")
    artifact_dir = Path(r"C:\Users\bhara\emotion-diary-ai\backend\artifacts")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    
    # Load processed data
    df = pd.read_csv(processed_path)

    # Fix NaNs -> replace with "" and make sure they're strings
    texts = df["clean_text"].fillna("").astype(str)
    
    # Fit TF-IDF
    vectorizer = TfidfVectorizer(max_features=1000, stop_words="english")
    vectorizer.fit(texts)
    
    # Save vectorizer
    vectorizer_path = artifact_dir / "tfidf_vectorizer.pkl"
    joblib.dump(vectorizer, vectorizer_path)
    
    print(f"✅ TF-IDF vectorizer saved at {vectorizer_path}")
    
if __name__ == "__main__":
    build_tfidf()
