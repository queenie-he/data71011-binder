# YouTube Maths-to-18 Discourse Analysis

This repository contains a fully reproducible Python pipeline to quantify public attitudes toward the UK 2023 proposal to extend compulsory mathematics education to age 18 (often dubbed *“Sunak’s Maths Plan”*).

## 1. Quick Start

```bash
# Create & activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# NLTK resources (one-off)
python - <<'PY'
import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')
PY

# Set your YouTube Data API key (replace XXX with your key)
export YOUTUBE_API_KEY="XXX"

# Run the pipeline (adjust flags as desired)
python youtube_math_analysis.py \
    --query "Sunak maths to 18" \
    --videos 25 \
    --comments 2000 \
    --topics 7
```

## 2. Outputs

```
outputs/
 ├── comments.csv               # Cleaned comments with sentiment & topic percentages
 ├── lda_topics.txt             # Top terms per discovered topic
 ├── decision_tree_rules.txt    # Interpretable decision-tree rules
 └── figures/
       ├── sentiment_distribution.png
       └── topic_importances.png
```

## 3. Methodology (Summary)

1. **Data Acquisition** – Searches YouTube for the supplied query, downloads top *N* videos and up to *K* comments per video via the official Data API.
2. **Pre-processing & Sentiment** – Cleans text (lower-case, tokenise, stop-word removal, lemmatisation) then scores sentiment with VADER, categorising as *positive*, *neutral* or *negative*.
3. **Topic Modelling** – Runs Latent Dirichlet Allocation to learn latent themes; saves topic keywords and assigns each comment a probability distribution over topics.
4. **Predictive Modelling** – Trains a DecisionTreeClassifier mapping topic probabilities → sentiment classes; exports both feature importances and human-readable decision rules.
5. **Reporting** – CSV of enriched comments plus publication-ready plots.

Feel free to customise any step (e.g. swap VADER for a transformer sentiment model, tune LDA hyper-parameters, or replace the tree with SHAP-interpretable gradient boosting). PRs welcome!
