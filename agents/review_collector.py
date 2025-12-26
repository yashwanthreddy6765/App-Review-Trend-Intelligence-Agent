import pandas as pd
import logging
from datetime import datetime, timedelta
from google_play_scraper import app, Sort, reviews

logger = logging.getLogger(__name__)

class ReviewCollector:
    """
    Agent responsible for collecting reviews from Google Play Store.
    Uses google-play-scraper library to fetch app reviews.
    """
    
    def __init__(self, app_id: str = 'com.whatsapp', n_reviews: int = 500):
        """
        Initialize ReviewCollector agent.
        
        Args:
            app_id: Google Play Store app ID
            n_reviews: Number of reviews to collect
        """
        self.app_id = app_id
        self.n_reviews = n_reviews
        logger.info(f'ReviewCollector initialized for app: {app_id}')
    
    def collect(self) -> pd.DataFrame:
        """
        Collect reviews from Google Play Store.
        
        Returns:
            DataFrame with columns: reviewId, content, score, at (date)
        """
        try:
            logger.info(f'Starting review collection for {self.app_id}...')
            
            # Fetch reviews
            result = [], None
            continuation_token = None
            collected = 0
            
            while collected < self.n_reviews:
                batch_result, continuation_token = reviews(
                    self.app_id,
                    lang='en',
                    country='us',
                    sort=Sort.NEWEST,
                    count=min(100, self.n_reviews - collected),
                    continuation_token=continuation_token
                )
                result[0].extend(batch_result)
                collected += len(batch_result)
                
                if not continuation_token:
                    break
            
            # Convert to DataFrame
            reviews_list = result[0][:self.n_reviews]
            
            df = pd.DataFrame([
                {
                    'reviewId': r.get('reviewId', ''),
                    'content': r.get('content', ''),
                    'score': r.get('score', 0),
                    'at': pd.to_datetime(r.get('at')),
                    'userName': r.get('userName', '')
                }
                for r in reviews_list
            ])
            
            # Filter to last 30 days
            cutoff_date = datetime.now() - timedelta(days=30)
            df['at'] = pd.to_datetime(df['at'])
            df = df[df['at'] >= cutoff_date].reset_index(drop=True)
            
            logger.info(f'Collected {len(df)} reviews (after filtering last 30 days)')
            return df
            
        except Exception as e:
            logger.error(f'Error collecting reviews: {e}')
            # Return empty DataFrame with correct structure
            return pd.DataFrame(columns=['reviewId', 'content', 'score', 'at', 'userName'])
