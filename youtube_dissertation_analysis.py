# youtube_dissertation_analysis.py
# -------------------------------------------------------------
# Dissertation Analysis Pipeline for YouTube Comments
# Topic: Audience Attitude Toward Sunak's Math Education Proposal
# -------------------------------------------------------------
# This script loads YouTube comments, performs text cleaning, sentiment analysis
# (VADER + TextBlob, averaged), visualizes sentiment, performs LDA topic modeling
# with coherence score visualization for k=2..10, fits multiple ML models to predict
# sentiment from topic probabilities, visualizes model performance, and prints best
# model and example comments for each topic. Detailed explanations are provided.
# -------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from gensim import corpora, models
from gensim.models import CoherenceModel
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

# Download NLTK resources if not already present
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')

# 1. Load Data
# -------------------------------------------------------------
df = pd.read_csv('youtube_comments_raw.csv')
print("Loaded data shape:", df.shape)
print(df.head())

# 2. Text Preprocessing
# -------------------------------------------------------------
# Clean text: remove URLs, punctuation, lowercase, remove stopwords, lemmatize
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text)
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'[^A-Za-z\s]', '', text)
    text = text.lower()
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Apply cleaning to comments
if 'comment' not in df.columns:
    raise ValueError("CSV must have a 'comment' column.")
df['clean_comment'] = df['comment'].apply(clean_text)
df = df[df['clean_comment'].str.strip() != '']
print("After cleaning:", df.shape)

# 3. Sentiment Analysis (VADER + TextBlob)
# -------------------------------------------------------------
sia = SentimentIntensityAnalyzer()

def vader_score(text):
    return sia.polarity_scores(text)['compound']

def textblob_score(text):
    return TextBlob(text).sentiment.polarity

# Compute both sentiment scores
print("Calculating sentiment scores...")
df['vader'] = df['clean_comment'].apply(vader_score)
df['textblob'] = df['clean_comment'].apply(textblob_score)
df['sentiment_score'] = df[['vader', 'textblob']].mean(axis=1)

# Assign sentiment label based on average score
# Thresholds: >=0.05 positive, <=-0.05 negative, else neutral
def label_sentiment(score):
    if score >= 0.05:
        return 'positive'
    elif score <= -0.05:
        return 'negative'
    else:
        return 'neutral'

df['sentiment'] = df['sentiment_score'].apply(label_sentiment)

# 4. Visualize Sentiment Distribution
# -------------------------------------------------------------
plt.figure(figsize=(8,4))
sns.histplot(df['sentiment_score'], bins=30, kde=True)
plt.title('Distribution of Averaged Sentiment Scores')
plt.xlabel('Averaged Sentiment Score (VADER + TextBlob)')
plt.ylabel('Number of Comments')
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(x='sentiment', data=df, palette='Set2')
plt.title('Sentiment Label Distribution')
plt.tight_layout()
plt.show()

# 5. Topic Modeling: Find Optimal k (number of topics)
# -------------------------------------------------------------
# Prepare data for LDA: tokenize
texts = [comment.split() for comment in df['clean_comment']]
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

# Try k from 2 to 10, compute coherence for each
coherence_scores = []
k_values = list(range(2, 11))
print("Calculating coherence scores for different k...")
for k in k_values:
    lda_model = models.LdaModel(corpus, num_topics=k, id2word=dictionary, passes=10, random_state=42)
    coherence_model = CoherenceModel(model=lda_model, texts=texts, dictionary=dictionary, coherence='c_v')
    coherence = coherence_model.get_coherence()
    coherence_scores.append(coherence)
    print(f'k={k}, Coherence={coherence:.3f}')

# Visualize coherence scores
plt.figure(figsize=(8,4))
plt.plot(k_values, coherence_scores, marker='o')
plt.xlabel('Number of Topics (k)')
plt.ylabel('Coherence Score')
plt.title('LDA Coherence Score vs Number of Topics')
plt.tight_layout()
plt.show()

# Choose best k (highest coherence)
best_k = k_values[np.argmax(coherence_scores)]
print(f"Best number of topics (k): {best_k}")

# Fit final LDA model
lda = models.LdaModel(corpus, num_topics=best_k, id2word=dictionary, passes=10, random_state=42)

# Show top words for each topic
print("\nTop words for each topic:")
for i in range(best_k):
    print(f"Topic {i}: {lda.print_topic(i)}")

# Assign topic probabilities to each comment
def get_topic_probs(text):
    bow = dictionary.doc2bow(text.split())
    topic_probs = lda.get_document_topics(bow, minimum_probability=0)
    return [prob for _, prob in topic_probs]

topic_probs = df['clean_comment'].apply(get_topic_probs)
topic_df = pd.DataFrame(topic_probs.tolist(), columns=[f'topic_{i}' for i in range(best_k)])
df = pd.concat([df.reset_index(drop=True), topic_df], axis=1)

# 6. Supervised Learning: Compare Models
# -------------------------------------------------------------
# Features: topic probabilities; Target: sentiment label
X = df[[f'topic_{i}' for i in range(best_k)]]
y = df['sentiment']

# Encode target labels numerically
y_encoded = LabelEncoder().fit_transform(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Define models to compare
models_dict = {
    'Decision Tree': DecisionTreeClassifier(max_depth=4, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'SVM': SVC(kernel='linear', probability=True, random_state=42)
}

results = {'Model': [], 'Accuracy': [], 'F1 Score': []}

print("\nTraining and evaluating models...")
for name, model in models_dict.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    results['Model'].append(name)
    results['Accuracy'].append(acc)
    results['F1 Score'].append(f1)
    print(f"{name}: Accuracy={acc:.3f}, F1={f1:.3f}")

# Visualize model performance
results_df = pd.DataFrame(results)
plt.figure(figsize=(8,5))
plt.plot(results_df['Model'], results_df['Accuracy'], marker='o', label='Accuracy')
plt.plot(results_df['Model'], results_df['F1 Score'], marker='s', label='F1 Score')
plt.title('Model Performance Comparison')
plt.ylabel('Score')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Print best model
best_idx = results_df['F1 Score'].idxmax()
print(f"\nBest model: {results_df.loc[best_idx, 'Model']} (F1 Score: {results_df.loc[best_idx, 'F1 Score']:.3f})")

# 7. Show example comments for each topic
# -------------------------------------------------------------
for i in range(best_k):
    top_comments = df.sort_values(f'topic_{i}', ascending=False).head(3)['comment']
    print(f"\nTop comments for Topic {i}:")
    for c in top_comments:
        print("-", c)

# End of script