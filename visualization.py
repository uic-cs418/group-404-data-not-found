import matplotlib.pyplot as plt
from textblob import TextBlob

def compute_sentiment(text_series):
    return text_series.apply(lambda x: TextBlob(x).sentiment.polarity)

def plot_sentiment_distribution(sentiments):
    plt.figure()
    plt.hist(sentiments, bins=30)
    plt.title("Sentiment Distribution of Articles")
    plt.xlabel("Sentiment Score")
    plt.ylabel("Frequency")
    plt.show()

def plot_articles_over_time(df):
    df['date'] = df['date'].astype(str).str[:10]
    counts = df.groupby('date').size()
    plt.figure()
    counts.plot()
    plt.title("Articles Over Time")
    plt.xlabel("Date")
    plt.ylabel("Count")
    plt.show()