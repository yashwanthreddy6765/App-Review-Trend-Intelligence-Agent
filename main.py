#!/usr/bin/env python
"""
Main orchestration script for Google Play Store Review Trend Analysis AI Agent.
This script coordinates all agents to collect, analyze, and deduplicate reviews.
"""

import os
import sys
import logging
from dotenv import load_dotenv
from agents.review_collector import ReviewCollector
from agents.topic_extractor import TopicExtractor
from agents.topic_deduplicator import TopicDeduplicator
from agents.trend_builder import TrendBuilder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def main():
    """
    Execute the complete pipeline:
    1. Collect reviews from Google Play Store
    2. Extract topics using LLM
    3. Deduplicate similar topics using embeddings
    4. Build trend table (T-30 to T)
    """
    
    # Configuration from environment
    app_id = os.getenv('GOOGLE_PLAY_APP_ID', 'com.whatsapp')
    trend_days = int(os.getenv('TREND_DAYS', '30'))
    openai_api_key = os.getenv('OPENAI_API_KEY')
    
    if not openai_api_key:
        logger.error('OPENAI_API_KEY not set in environment')
        sys.exit(1)
    
    logger.info(f'Starting pipeline for app: {app_id}')
    logger.info(f'Trend window: {trend_days} days')
    
    # Create output directories
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('output', exist_ok=True)
    
    # Step 1: Collect Reviews
    logger.info('Step 1: Collecting reviews...')
    collector = ReviewCollector(app_id=app_id, n_reviews=500)
    reviews_df = collector.collect()
    reviews_path = f'data/raw/reviews_{app_id}.csv'
    reviews_df.to_csv(reviews_path, index=False)
    logger.info(f'Collected {len(reviews_df)} reviews -> {reviews_path}')
    
    # Step 2: Extract Topics
    logger.info('Step 2: Extracting topics from reviews...')
    extractor = TopicExtractor(api_key=openai_api_key, model='gpt-3.5-turbo')
    topics_df = extractor.extract(reviews_df)
    topics_path = f'data/processed/topics_{app_id}.csv'
    topics_df.to_csv(topics_path, index=False)
    logger.info(f'Extracted {len(topics_df)} topics -> {topics_path}')
    
    # Step 3: Deduplicate Topics
    logger.info('Step 3: Deduplicating topics using embeddings...')
    deduplicator = TopicDeduplicator(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        threshold=0.85
    )
    dedup_df = deduplicator.deduplicate(topics_df)
    dedup_path = f'data/processed/topics_dedup_{app_id}.csv'
    dedup_df.to_csv(dedup_path, index=False)
    logger.info(f'Deduplicated to {len(dedup_df["canonical_topic"].unique())} unique topics -> {dedup_path}')
    
    # Step 4: Build Trend Table
    logger.info('Step 4: Building trend table...')
    builder = TrendBuilder(days=trend_days)
    trend_df = builder.build(dedup_df)
    trend_path = f'output/trend_{app_id}.csv'
    trend_df.to_csv(trend_path)
    logger.info(f'Trend table created: {trend_path}')
    logger.info(f'\nTrend Table Summary:')
    logger.info(f'Topics: {len(trend_df)}')
    logger.info(f'Date range: {trend_days} days')
    
    logger.info('Pipeline completed successfully!')
    return trend_path

if __name__ == '__main__':
    try:
        output_file = main()
        print(f'\nFinal output: {output_file}')
    except Exception as e:
        logger.error(f'Pipeline failed: {e}', exc_info=True)
        sys.exit(1)
