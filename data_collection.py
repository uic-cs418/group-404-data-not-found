import requests
import pandas as pd

def fetch_gdelt_data(keywords, max_records=1000):
    base_url = "https://api.gdeltproject.org/api/v2/doc/doc"
    all_articles = []

    for keyword in keywords:
        params = {
            "query": keyword,
            "mode": "ArtList",
            "maxrecords": max_records,
            "format": "json"
        }

        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            data = response.json()

            if "articles" in data:
                for article in data["articles"]:
                    all_articles.append({
                        "title": article.get("title"),
                        "text": article.get("title", ""),
                        "source": article.get("sourcecountry"),
                        "date": article.get("seendate")
                    })

    return pd.DataFrame(all_articles)

def fetch_newsapi_data(api_key, keywords, page_size=100):
    url = "https://newsapi.org/v2/everything"
    all_articles = []

    for keyword in keywords:
        params = {
            "q": keyword,
            "language": "en",
            "pageSize": page_size,
            "apiKey": api_key
        }

        response = requests.get(url, params=params)
        if response.status_code == 200:
            articles = response.json().get("articles", [])

            for a in articles:
                all_articles.append({
                    "title": a["title"],
                    "text": a["content"] if a["content"] else a["description"],
                    "source": a["source"]["name"],
                    "date": a["publishedAt"]
                })

    return pd.DataFrame(all_articles)