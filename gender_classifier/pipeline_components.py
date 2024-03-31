import pandas as pd
import re
import logging
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import resample
from spacy.language import Language

import logging_config

logging_config.configure_logging()
LOGGER = logging.getLogger(__name__)


class DropAndTransformHandler(BaseEstimator, TransformerMixin):
    def __init__(self, column_name: str):
        self.column_name = column_name

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        LOGGER.info(f'NOW PERFORMING DROP AND TRANSFORM FOR {X.shape[0]} ROWS')
        X[self.column_name] = X.filter(like='essay').apply(lambda row: '\n'.join(row.dropna()), axis=1)
        columns_to_drop = set(X.columns) - {self.column_name, 'sex'}
        X = X.drop(columns=columns_to_drop)
        X = X[X[self.column_name] != '']

        LOGGER.info(f'PROCESS COMPLETE FOR {X.shape[0]} ROWS')
        return X


class TextPatternCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, column_name: str):
        self.url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        self.html_pattern = r'<.*?>+'
        self.digit_pattern = r'\d'
        self.new_line = r'\n'
        self.multiple_spaces = r'\s+'
        self.column_name = column_name

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        LOGGER.info(f'NOW PERFORMING PATTERN CLEANING FOR {X.shape[0]} ROWS')
        X[self.column_name] = X[self.column_name].astype(str).apply(self.remove_patterns)

        LOGGER.info(f'PROCESS COMPLETE FOR {X.shape[0]} ROWS')
        return X

    def remove_patterns(self, text: str) -> str:
        text = re.sub(self.url_pattern, '', text)
        text = re.sub(self.html_pattern, '', text)
        text = re.sub(self.digit_pattern, '', text)
        text = re.sub(self.new_line, ' ', text)
        text = re.sub(self.multiple_spaces, ' ', text)

        return text


class TextNormalizer(BaseEstimator, TransformerMixin):
    def __init__(self, column_name: str, nlp: Language) -> None:
        self.column_name = column_name
        self.nlp = nlp

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        LOGGER.info(f'NOW PERFORMING SPACY NORMALIZATION FOR {X.shape[0]} ROWS')
        X[self.column_name] = X[self.column_name].astype(str).apply(self.preprocess_text)

        LOGGER.info(f'PROCESS COMPLETE FOR {X.shape[0]} ROWS')
        return X

    def preprocess_text(self, text: str) -> str:
        doc = self.nlp(text)
        return " ".join([token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct])


class DataResampler(BaseEstimator, TransformerMixin):
    def __init__(self, target_column):
        self.target_column = target_column

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        LOGGER.info(f'NOW PERFORMING DATA RESAMPLING FOR {X.shape[0]} ROWS')
        gender_counts = X[self.target_column].value_counts()
        minority_gender = gender_counts.idxmin()
        minority_count = gender_counts.min()
        majority_count = gender_counts.max()

        df_majority = X[X[self.target_column] != minority_gender]
        df_minority = X[X[self.target_column] == minority_gender]

        df_oversampled = resample(df_minority, replace=True, n_samples=majority_count, random_state=42)
        df_resampled = pd.concat([df_oversampled, df_majority])

        LOGGER.info(f'PROCESS COMPLETE: {df_resampled.shape[0]} ROWS')
        return df_resampled.sample(frac=1, random_state=42)


class DropNullRows(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        LOGGER.info(f'NOW DROPPING ROWS CONTAINING NULL VALUES FOR {X.shape[0]} ROWS')
        return X.dropna()

