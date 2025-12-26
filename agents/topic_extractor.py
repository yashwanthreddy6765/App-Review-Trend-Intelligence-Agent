import pandas as pd
import logging
import json
from typing import List, Dict
import openai

logger = logging.getLogger(__name__)

class TopicExtractor:
    """
    Agent responsible for extracting structured topics from reviews using LLM.
    Uses OpenAI's API to identify issues, feature requests, and feedback.
    """
    
    def __init__(self, api_key: str, model: str = 'gpt-3.5-turbo'):
        """
        Initialize TopicExtractor agent.
        
        Args:
            api_key: OpenAI API key
            model: Model to use for extraction
        """
        openai.api_key = api_key
        self.model = model
        logger.info(f'TopicExtractor initialized with model: {model}')
    
    def extract(self, reviews_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract topics from review texts using LLM.
        
        Args:
            reviews_df: DataFrame with review data
            
        Returns:
            DataFrame with extracted topics
        """
        topics = []
        
        for idx, row in reviews_df.iterrows():
            try:
                topic_record = self._extract_from_review(
                    review_id=row['reviewId'],
                    review_text=row['content'],
                    date=row['at']
                )
                if topic_record:
                    topics.append(topic_record)
            except Exception as e:
                logger.warning(f'Error extracting topic from review {row["reviewId"]}: {e}')
        
        df = pd.DataFrame(topics)
        logger.info(f'Extracted {len(df)} topics from {len(reviews_df)} reviews')
        return df
    
    def _extract_from_review(self, review_id: str, review_text: str, date) -> Dict:
        """
        Extract topic from a single review.
        """
        if not review_text or len(review_text.strip()) == 0:
            return None
        
        prompt = f"""
Analyze this app review and extract the main topic.
Respond with a JSON object with these fields:
- category: one of ['issue', 'feature_request', 'feedback', 'other']
- topic: brief description of the topic
- sentiment: one of ['positive', 'negative', 'neutral']

Review: {review_text}

JSON Response:
"""
        
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=100
            )
            
            response_text = response['choices'][0]['message']['content'].strip()
            # Parse JSON response
            try:
                data = json.loads(response_text)
            except json.JSONDecodeError:
                # Fallback parsing
                data = {
                    'category': 'feedback',
                    'topic': review_text[:50],
                    'sentiment': 'neutral'
                }
            
            return {
                'review_id': review_id,
                'date': date,
                'category': data.get('category', 'other'),
                'topic': data.get('topic', review_text[:50]),
                'sentiment': data.get('sentiment', 'neutral')
            }
        
        except Exception as e:
            logger.error(f'OpenAI API error: {e}')
            return {
                'review_id': review_id,
                'date': date,
                'category': 'feedback',
                'topic': review_text[:50],
                'sentiment': 'neutral'
            }
