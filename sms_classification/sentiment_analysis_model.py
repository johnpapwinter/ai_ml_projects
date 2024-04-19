import pandas as pd
import warnings
from transformers import pipeline

from pipeline_components import *


class SentimentModeler:
    def __init__(self, task='sentiment-analysis', model_name='distilbert-base-uncased-finetuned-sst-2-english'):
        warnings.filterwarnings('ignore', category=FutureWarning)
        self.model_name = model_name
        self.task = task
        self.classifier = pipeline(self.task, model=self.model_name)

    def assign_sentiment(self, df: pd.DataFrame, text_column: str, sentiment_column: str) -> pd.DataFrame:
        df[sentiment_column] = df[text_column].apply(self.__analyze_sentiment)
        return df

    def __analyze_sentiment(self, text: str) -> str:
        result = self.classifier(text)
        return result[0]['label']


