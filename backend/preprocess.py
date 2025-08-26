# FastAPI backend entry point 
import pandas as pd 
import re
from pathlib import Path
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

def clean_text(text:str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]","",text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in ENGLISH_STOP_WORDS]
    return "".join(tokens)

def preprocess():
    raw_path = Path(r"C:\Users\bhara\emotion-diary-ai\data\raw\Combined Data.csv")
    processed_path = Path(r"C:\Users\bhara\emotion-diary-ai\data\processed\combined_data_clean.csv")
    
    df = pd.read_csv(raw_path)
    df["clean_text"] = df["statement"].apply(clean_text)
    
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_path, index=False)
    print(f"Processed data saved at{processed_path}")
    
if __name__ == "__main__":
    preprocess()