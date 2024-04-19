import spacy
import logging
from sklearn.pipeline import Pipeline

from pipeline_components import *

import logging_config

logging_config.configure_logging()
LOGGER = logging.getLogger(__name__)


def dominant_sentiment_by_topic(df: pd.DataFrame) -> pd.DataFrame:
    topic_sentiment_counts = df.groupby(['Topic', 'Sentiment']).size().reset_index(name='count')

    dominant_sentiments = topic_sentiment_counts.loc[topic_sentiment_counts.groupby('Topic')['count'].idxmax()]

    return dominant_sentiments


class DataPreprocessor:
    def __init__(self, column_name):
        self.column_name = column_name
        self.nlp = spacy.load('en_core_web_sm')

    def preprocess_data(self) -> Pipeline:
        return Pipeline(steps=[
            ("drop_handler", DropColumnHandler(self.column_name)),
            ("slang_mapper", SlangWordsMapper(self.column_name)),
            ("noise_remover", TextNoiseRemover(self.column_name)),
        ])

    def add_normalized_text(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        LOGGER.info("Adding spaCy normalized text column")
        df[column] = df[column].astype(str).apply(self.normalize_text)
        return df

    def normalize_text(self, text: str) -> str:
        doc = self.nlp(text)
        return " ".join([token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct])

