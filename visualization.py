import matplotlib.pyplot as plt
from textblob import TextBlob

def compute_sentiment(text_series):
    return text_series.apply(lambda x: TextBlob(str(x)).sentiment.polarity)

def plot_sentiment_distribution(sentiments):
    plt.figure(figsize=(8, 5))
    plt.hist(sentiments, bins=30)
    plt.title("Sentiment Distribution of Articles")
    plt.xlabel("Sentiment Score")
    plt.ylabel("Frequency")
    plt.show()

def plot_articles_over_time(df):
    df = df.copy()
    df['date'] = df['date'].astype(str).str[:10]
    counts = df.groupby('date').size()

    plt.figure(figsize=(10, 5))
    counts.plot()
    plt.title("Articles Over Time")
    plt.xlabel("Date")
    plt.ylabel("Count")
    plt.show()

def plot_top_sources(df, top_n=10):
    plt.figure(figsize=(10, 5))
    df["source"].value_counts().head(top_n).plot(kind="bar")
    plt.title(f"Top {top_n} News Sources")
    plt.xlabel("Source")
    plt.ylabel("Article Count")
    plt.xticks(rotation=45)
    plt.show()