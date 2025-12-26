import pandas as pd
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class TrendBuilder:
    """
    Agent responsible for building trend tables from deduplicated topics.
    Creates a pivot table with topics as rows and dates as columns (T-N to T).
    """
    
    def __init__(self, days: int = 30):
        """
        Initialize TrendBuilder agent.
        
        Args:
            days: Number of days to look back (e.g., 30 for T-30 to T)
        """
        self.days = days
        logger.info(f'TrendBuilder initialized for {days} days')
    
    def build(self, topics_df: pd.DataFrame) -> pd.DataFrame:
        """
        Build trend table from deduplicated topics.
        
        Args:
            topics_df: DataFrame with deduplicated topics and dates
            
        Returns:
            DataFrame with topics as rows and dates as columns
        """
        if len(topics_df) == 0:
            logger.warning('Empty DataFrame provided')
            return pd.DataFrame()
        
        # Ensure date column is datetime
        topics_df['date'] = pd.to_datetime(topics_df['date'])
        
        # Filter to last N days
        cutoff_date = datetime.now() - timedelta(days=self.days)
        filtered_df = topics_df[topics_df['date'] >= cutoff_date].copy()
        
        # Extract date (without time)
        filtered_df['date_only'] = filtered_df['date'].dt.date
        
        # Aggregate: count topics per canonical_topic per date
        trend_data = filtered_df.groupby(['canonical_topic', 'date_only']).size().reset_index(name='count')
        
        # Pivot: topics as rows, dates as columns
        trend_pivot = trend_data.pivot_table(
            index='canonical_topic',
            columns='date_only',
            values='count',
            aggfunc='sum',
            fill_value=0
        )
        
        # Convert to integers
        trend_pivot = trend_pivot.astype(int)
        
        # Reset index to make canonical_topic a column
        trend_pivot.reset_index(inplace=True)
        
        logger.info(f'Built trend table: {len(trend_pivot)} topics x {len(trend_pivot.columns)-1} days')
        return trend_pivot
