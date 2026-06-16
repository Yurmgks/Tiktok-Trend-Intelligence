# TikTok Trend Intelligence & Social Listening Dashboard

This is a professional-grade TikTok Trend Intelligence & Social Listening Dashboard built with Python, Streamlit, and Machine Learning libraries. It simulates and runs a full social media monitoring architecture, covering data collection, database ingestion, NLP preprocessing, Indonesian sentiment/emotion analysis, topic modeling, viral prediction, and trend forecasting.

## Architecture

1. **Data Collection**: Playwright-based scraper templates for videos, comments, music, hashtags, and creators. Includes a live mock data generator to bootstrap analysis.
2. **Database Layer**: SQLAlchemy-based database manager supporting **PostgreSQL** with an automatic **SQLite fallback** (`tiktok_trends.db`) for portable execution.
3. **NLP Preprocessing**: Case folding, cleaning, tokenization, Indonesian stopword removal, and stemming using `Sastrawi` and `NLTK`.
4. **Sentiment & Emotion Analysis**: Indonesian classification mapping comments to Positive/Neutral/Negative (using IndoBERT/IndoBERTweet models/lexicons) and emotions to *Joy, Anger, Sadness, Fear, Surprise, and Love*.
5. **Topic Modeling**: Clustering of text into automatic topics using a semantic modeling pipeline.
6. **Trend Engine**: Calculates a custom Trend Score:
   $$\text{Trend Score} = (\text{Views Growth} \times 0.4) + (\text{Comments Growth} \times 0.3) + (\text{Shares Growth} \times 0.3)$$
   Provides 7-day future forecasting and XGBoost/Random Forest-based viral prediction models.
7. **Streamlit Dashboard**: A dashboard with charts, comparative sentiment analysis (TikTok vs YouTube), and interactive viral predictor tool.

## Directory Structure

```
tiktok-trend-intelligence/
├── scraper/             # Scraper templates and mock data generator
├── database/            # SQLAlchemy database connection and models
├── preprocessing/       # Indonesian NLP cleaning and stemming
├── sentiment/           # IndoBERT-aligned sentiment analysis
├── emotion/             # Emotion classification mapping
├── topic_modeling/      # Auto topic modeling
├── prediction/          # Trend forecasting and viral prediction models
├── dashboard/           # Streamlit application entry point
├── requirements.txt     # Python dependencies
└── README.md            # Technical documentation
```

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Database Ingestion
Initialize the database and load simulation data:
```bash
python scraper/scraper.py
```

### 3. Start the Dashboard
```bash
streamlit run dashboard/app.py
```
