import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

from pipeline_components import *


class TopicModeler:
    def __init__(self, n_topics=5):
        self.n_topics = n_topics
        self.lda = LatentDirichletAllocation(n_components=self.n_topics, random_state=42)
        self.vectorizer = TfidfVectorizer()

    def __vectorize(self, df: pd.DataFrame, text_column: str) -> np.array:
        return self.vectorizer.fit_transform(df[text_column])

    def fit(self, df: pd.DataFrame, text_column: str) -> None:
        text_matrix = self.__vectorize(df, text_column)
        self.lda.fit(text_matrix)

    def get_topics(self) -> list:
        topic_list = []
        for i, topic in enumerate(self.lda.components_):
            topic_list.append(" ".join(self.vectorizer.get_feature_names_out()[j] for j in topic.argsort()[-10:]))
        return topic_list

    def assign_topics(self, df: pd.DataFrame, text_column: str, topic_column: str) -> pd.DataFrame:
        text_matrix = self.__vectorize(df, text_column)

        topic_distributions = self.lda.transform(text_matrix)
        assigned_topics = np.argmax(topic_distributions, axis=1)
        df[topic_column] = assigned_topics

        return df

    def save_model(self, filename: str) -> None:
        joblib.dump(self.lda, filename)

    def load_model(self, filename: str) -> None:
        self.lda = joblib.load(filename)


