import pandas as pd
import numpy as np
import logging
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple

logger = logging.getLogger(__name__)

class TopicDeduplicator:
    """
    Agent responsible for deduplicating similar topics using embeddings.
    Uses sentence transformers to encode topics and cosine similarity to find duplicates.
    """
    
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2', threshold: float = 0.85):
        """
        Initialize TopicDeduplicator agent.
        
        Args:
            model_name: Sentence transformer model to use
            threshold: Similarity threshold for considering topics as duplicates (0-1)
        """
        self.model = SentenceTransformer(model_name)
        self.threshold = threshold
        logger.info(f'TopicDeduplicator initialized with model: {model_name}, threshold: {threshold}')
    
    def deduplicate(self, topics_df: pd.DataFrame) -> pd.DataFrame:
        """
        Deduplicate topics using embeddings and cosine similarity.
        
        Args:
            topics_df: DataFrame with topic data
            
        Returns:
            DataFrame with added canonical_topic column
        """
        if len(topics_df) == 0:
            logger.warning('Empty DataFrame provided')
            topics_df['canonical_topic'] = []
            return topics_df
        
        # Get unique topics
        unique_topics = topics_df['topic'].unique().tolist()
        logger.info(f'Found {len(unique_topics)} unique topics')
        
        # Encode topics
        embeddings = self.model.encode(unique_topics, show_progress_bar=False)
        
        # Compute similarity matrix
        similarity_matrix = cosine_similarity(embeddings)
        
        # Greedy clustering
        canonical_map = self._greedy_cluster(unique_topics, similarity_matrix)
        
        # Map topics to canonical topics
        topics_df['canonical_topic'] = topics_df['topic'].map(canonical_map)
        
        n_canonical = len(set(canonical_map.values()))
        logger.info(f'Deduplicated {len(unique_topics)} topics to {n_canonical} canonical topics')
        
        return topics_df
    
    def _greedy_cluster(self, topics: List[str], similarity_matrix: np.ndarray) -> dict:
        """
        Greedy clustering: assign each topic to closest canonical topic or create new one.
        
        Args:
            topics: List of topic strings
            similarity_matrix: Similarity matrix between topics
            
        Returns:
            Dictionary mapping topic -> canonical_topic
        """
        canonical_map = {}
        canonical_topics = []
        canonical_indices = []
        
        for i, topic in enumerate(topics):
            if topic in canonical_map:
                continue
            
            # Check similarity with existing canonical topics
            found_canonical = False
            for c_idx in canonical_indices:
                sim = similarity_matrix[i, c_idx]
                if sim >= self.threshold:
                    canonical_map[topic] = canonical_topics[canonical_indices.index(c_idx)]
                    found_canonical = True
                    break
            
            # If no similar canonical topic, create new one
            if not found_canonical:
                canonical_map[topic] = topic
                canonical_topics.append(topic)
                canonical_indices.append(i)
        
        return canonical_map
