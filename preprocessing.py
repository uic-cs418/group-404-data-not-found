import re
import pandas as pd
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

def clean_text(text):
    if pd.isna(text):
        return ""

    text = text.lower()
    text = re.sub(r'http\S+', '', text)  # remove URLs
    text = re.sub(r'[^a-z\s]', '', text)  # remove punctuation/numbers
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

def preprocess_dataframe(df):
    print("Dataset shape before cleaning:", df.shape)

    df = df.drop_duplicates()
    df = df.dropna(subset=["text"])

    df["clean_text"] = df["text"].apply(clean_text)

    print("Dataset shape after cleaning:", df.shape)
    return df