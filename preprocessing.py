import re
import pandas as pd
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

def preprocess_dataframe(df):
    df = df.drop_duplicates()
    df = df.dropna(subset=["text"])

    df["clean_text"] = df["text"].apply(clean_text)
    return df