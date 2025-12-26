# Google Play Store Review Trend Analysis AI Agent

## Problem Statement

Product teams need to understand review trends over time to identify emerging issues, feature requests, and user sentiment patterns. Manual analysis is time-consuming and doesn't scale. This project builds an intelligent agent-based system that automatically collects, analyzes, and deduplicates topics from Google Play Store reviews across a 30-day window.

## Agentic Architecture

The system uses a modular agent-based design with four specialized agents:

### 1. ReviewCollector Agent
- **Role**: Fetches reviews from Google Play Store using `google-play-scraper`
- **Output**: Raw reviews with metadata (ID, content, score, date)
- **File**: `agents/review_collector.py`

### 2. TopicExtractor Agent
- **Role**: Uses OpenAI's LLM to extract structured topics from reviews
- **Extracts**: Issues, feature requests, and feedback with sentiment
- **Output**: Structured topic records with categories
- **File**: `agents/topic_extractor.py`

### 3. TopicDeduplicator Agent
- **Role**: Deduplicates similar topics using sentence embeddings and cosine similarity
- **Method**: Greedy clustering with similarity threshold
- **Output**: Canonical topics with cluster assignments
- **File**: `agents/topic_deduplicator.py`

### 4. TrendBuilder Agent
- **Role**: Aggregates deduplicated topics by date
- **Output**: Trend table showing topic frequency across days (T-30 to T)
- **File**: `agents/trend_builder.py`

## Topic Deduplication Approach

Instead of traditional LDA or classical topic modeling, we use:

1. **Sentence Embeddings**: Convert topics to dense vectors using `sentence-transformers` (all-MiniLM-L6-v2)
2. **Cosine Similarity**: Compute pairwise similarity between topic embeddings
3. **Greedy Clustering**: Assign similar topics to canonical clusters
4. **Threshold**: Topics with similarity > 0.85 are considered duplicates

## How to Run

### Prerequisites
```bash
pip install -r requirements.txt
```

### Environment Setup
Create a `.env` file:
```
OPENAI_API_KEY=your_api_key_here
GOOGLE_PLAY_APP_ID=com.whatsapp  # Optional, defaults to WhatsApp
TREND_DAYS=30  # Optional, defaults to 30
```

### Run the Pipeline
```bash
python main.py
```

### Output
- **Raw reviews**: `data/raw/reviews_{app_id}.csv`
- **Extracted topics**: `data/processed/topics_{app_id}.csv`
- **Deduplicated topics**: `data/processed/topics_dedup_{app_id}.csv`
- **Trend table**: `output/trend_{app_id}.csv`

## Limitations

1. **LLM Cost**: OpenAI API calls can be expensive for large review volumes
2. **Scraper Constraints**: google-play-scraper has rate limits and may not fetch all reviews
3. **Simple Clustering**: Greedy clustering is fast but may miss some duplicates
4. **Count-Based Trends**: Only tracks frequency, not sentiment intensity
5. **Language Support**: Topic extraction works best for English reviews
6. **Temporal Gaps**: Missing reviews on days with no new content

## Project Structure

```
App-Review-Trend-Intelligence-Agent/
├── agents/
│   ├── review_collector.py
│   ├── topic_extractor.py
│   ├── topic_deduplicator.py
│   └── trend_builder.py
├── data/
│   ├── raw/
│   └── processed/
├── output/
├── main.py
├── requirements.txt
└── README.md
```

## Sample Output

The `output/sample_trend_output.csv` shows a typical trend table with topics as rows and dates as columns:

```
canonical_topic,2025-12-21,2025-12-22,2025-12-23,...
login issues,5,8,12,...
notification delay,3,4,7,...
feature request: dark mode,2,3,5,...
```

## Dependencies

- **pandas**: Data manipulation and CSV handling
- **numpy**: Numerical operations
- **google-play-scraper**: Fetch reviews from Google Play Store
- **sentence-transformers**: Generate embeddings for topic deduplication
- **scikit-learn**: Cosine similarity computation
- **openai**: LLM-based topic extraction
- **python-dotenv**: Environment variable management
