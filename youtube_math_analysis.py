#!/usr/bin/env python
"""
youtube_math_analysis.py
================================================
Pipeline to quantify public discourse surrounding the UK 2023 proposal
to extend compulsory mathematics education to age 18 ("Sunak Math Plan").

Steps
-----
1.  Data Acquisition: search YouTube, fetch comments & basic video metadata
2.  NLP Pre-processing: clean, tokenize, lemmatise, remove stop-words
3.  Sentiment Analysis: VADER (positive / neutral / negative + compound score)
4.  Topic Modelling: Latent Dirichlet Allocation (Gensim) to learn latent themes
5.  Predictive Modelling: DecisionTree linking topics → sentiment
6.  Interpretation: feature importances & human-readable rules

Requirements
------------
See `requirements.txt`.
Set `YOUTUBE_API_KEY` environment variable with a valid key.

Usage (after installing requirements):
$ export YOUTUBE_API_KEY="<YOUR_KEY>"
$ python youtube_math_analysis.py --query "Sunak maths until 18" --videos 25 --comments 2000

Outputs
-------
- comments.csv            : raw comments with sentiment & topic distribution
- lda_topics.txt          : top terms per discovered topic
- decision_tree_rules.txt : interpretable rules mapping topics → sentiment
- figures/                : sentiment distribution + topic importance plots
"""
import os
import re
import argparse
import csv
from collections import defaultdict
from typing import List, Tuple

import pandas as pd
import numpy as np
from tqdm import tqdm

# YouTube API
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# NLP
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Topic modelling
import gensim
from gensim import corpora

# ML
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# Vis
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure NLTK resources are available
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

STOP_WORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()
SIA = SentimentIntensityAnalyzer()


def clean_text(text: str) -> str:
    """Lowercase, remove URLs, punctuation, digits, and lemmatise."""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)  # URLs
    text = re.sub(r"[^a-z\s]", " ", text)  # keep letters & spaces
    tokens = [LEMMATIZER.lemmatize(tok) for tok in text.split() if tok not in STOP_WORDS and len(tok) > 2]
    return " ".join(tokens)


def sentiment_category(score: float) -> str:
    if score >= 0.05:
        return "positive"
    elif score <= -0.05:
        return "negative"
    return "neutral"


def youtube_service(api_key: str):
    return build("youtube", "v3", developerKey=api_key)


def search_video_ids(service, query: str, max_results: int) -> List[str]:
    resp = service.search().list(
        q=query,
        part="id",
        type="video",
        maxResults=min(max_results, 50),
        relevanceLanguage="en",
        order="relevance",
    ).execute()
    return [item["id"]["videoId"] for item in resp.get("items", [])]


def fetch_comments_for_video(service, video_id: str, max_comments: int) -> List[Tuple[str, int]]:
    """Return list of (commentText, likeCount)."""
    comments = []
    next_token = None
    while len(comments) < max_comments:
        try:
            resp = service.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=100,
                pageToken=next_token,
                textFormat="plainText",
            ).execute()
        except HttpError as e:
            print(f"⚠️  YouTube API error on video {video_id}: {e}")
            break
        for item in resp.get("items", []):
            snippet = item["snippet"]["topLevelComment"]["snippet"]
            comments.append((snippet.get("textDisplay", ""), snippet.get("likeCount", 0)))
            if len(comments) >= max_comments:
                break
        next_token = resp.get("nextPageToken")
        if not next_token:
            break
    return comments


def download_comments(api_key: str, query: str, max_videos: int, max_comments_per_video: int) -> pd.DataFrame:
    service = youtube_service(api_key)
    video_ids = search_video_ids(service, query, max_results=max_videos)
    rows = []
    for vid in tqdm(video_ids, desc="Fetching videos"):
        for text, likes in fetch_comments_for_video(service, vid, max_comments_per_video):
            rows.append({"video_id": vid, "raw_comment": text, "like_count": likes})
    df = pd.DataFrame(rows)
    return df


def preprocess_and_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = []
    sentiments = []
    categories = []
    for txt in tqdm(df["raw_comment"], desc="Pre-processing & sentiment"):
        ctext = clean_text(txt)
        cleaned.append(ctext)
        score = SIA.polarity_scores(ctext)["compound"]
        sentiments.append(score)
        categories.append(sentiment_category(score))
    df["clean_comment"] = cleaned
    df["sentiment_score"] = sentiments
    df["sentiment_cat"] = categories
    return df


def lda_topics(texts: List[str], num_topics: int = 7, passes: int = 10):
    token_lists = [t.split() for t in texts]
    dictionary = corpora.Dictionary(token_lists)
    dictionary.filter_extremes(no_below=5, no_above=0.4)
    corpus = [dictionary.doc2bow(tokens) for tokens in token_lists]
    lda_model = gensim.models.LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        passes=passes,
        random_state=42,
    )
    return lda_model, corpus, dictionary


def topic_distributions(lda_model, corpus, num_topics: int) -> np.ndarray:
    dist = np.zeros((len(corpus), num_topics))
    for i, doc in enumerate(corpus):
        for topic_id, prob in lda_model.get_document_topics(doc):
            dist[i, topic_id] = prob
    return dist


def train_decision_tree(X: np.ndarray, y: List[str]):
    clf = DecisionTreeClassifier(max_depth=4, random_state=42)
    clf.fit(X, y)
    return clf


def plot_sentiment_distribution(df: pd.DataFrame, outdir: str):
    plt.figure(figsize=(6,4))
    sns.countplot(x="sentiment_cat", data=df, order=["positive", "neutral", "negative"], palette="Set2")
    plt.title("Sentiment Distribution")
    plt.xlabel("")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "sentiment_distribution.png"))
    plt.close()


def plot_topic_importance(clf, outdir: str):
    plt.figure(figsize=(8,4))
    sns.barplot(x=list(range(len(clf.feature_importances_))), y=clf.feature_importances_, color="skyblue")
    plt.title("Decision Tree Topic Importances")
    plt.xlabel("Topic #")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "topic_importances.png"))
    plt.close()


def save_topics(lda_model, num_words: int, filepath: str):
    with open(filepath, "w", encoding="utf-8") as f:
        for tid, words in lda_model.show_topics(num_topics=-1, num_words=num_words):
            f.write(f"Topic {tid}: {words}\n")


def main():
    parser = argparse.ArgumentParser(description="YouTube Maths-to-18 discourse analysis")
    parser.add_argument("--query", default="Sunak maths to 18", help="YouTube search query")
    parser.add_argument("--videos", type=int, default=20, help="Number of videos to fetch")
    parser.add_argument("--comments", type=int, default=1000, help="Approx max comments per video")
    parser.add_argument("--topics", type=int, default=7, help="Number of LDA topics")
    parser.add_argument("--outdir", default="outputs", help="Directory to store artifacts")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(os.path.join(args.outdir, "figures"), exist_ok=True)

    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        raise RuntimeError("Please set YOUTUBE_API_KEY environment variable.")

    print("\n==== 1. Downloading comments ====")
    df = download_comments(api_key, args.query, args.videos, args.comments)
    print(f"Fetched {len(df)} comments from {df.video_id.nunique()} videos")

    print("\n==== 2. Cleaning & Sentiment analysis ====")
    df = preprocess_and_sentiment(df)

    print("\n==== 3. Topic modelling ====")
    lda_model, corpus, dictionary = lda_topics(df.clean_comment.tolist(), num_topics=args.topics)
    topic_dist = topic_distributions(lda_model, corpus, num_topics=args.topics)
    topic_cols = [f"topic_{i}" for i in range(args.topics)]
    df[topic_cols] = topic_dist

    print("\n==== 4. Decision tree ====")
    clf = train_decision_tree(topic_dist, df.sentiment_cat.tolist())
    rules_text = export_text(clf, feature_names=topic_cols)
    with open(os.path.join(args.outdir, "decision_tree_rules.txt"), "w") as f:
        f.write(rules_text)
    print("Decision tree rules written to decision_tree_rules.txt")

    print("\n==== 5. Saving artifacts ====")
    df.to_csv(os.path.join(args.outdir, "comments.csv"), index=False)
    save_topics(lda_model, 15, os.path.join(args.outdir, "lda_topics.txt"))

    print("\n==== 6. Visualisations ====")
    plot_sentiment_distribution(df, os.path.join(args.outdir, "figures"))
    plot_topic_importance(clf, os.path.join(args.outdir, "figures"))
    print("Figures saved to outputs/figures/")

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()