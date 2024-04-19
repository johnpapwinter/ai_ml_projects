import sys

import pandas as pd
import re
import logging
from sklearn.base import BaseEstimator, TransformerMixin
from spacy.language import Language

from slang_replacement_map import word_mapper

logger = logging.getLogger(__name__)
logging.basicConfig(handlers=[logging.StreamHandler(sys.stdout)],
                    format='[%(asctime)s : %(levelname)s : %(message)s]',
                    level=logging.DEBUG)


class DropColumnHandler(BaseEstimator, TransformerMixin):
    def __init__(self, column_name: str) -> None:
        self.__column_name = column_name

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        logger.info('DROPPING UNNEEDED COLUMNS')
        columns_to_drop = set(X.columns) - {self.__column_name}
        X = X.drop(columns=columns_to_drop)
        X = X[X[self.__column_name] != '']

        logger.info('PROCESS COMPLETE')
        return X


class SlangWordsMapper(BaseEstimator, TransformerMixin):
    def __init__(self, column_name: str):
        self.__column_name = column_name

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        logger.info('REPLACING SLANG WITH WORDS')
        X[self.__column_name] = X[self.__column_name].astype(str).apply(self.replace_words)

        logger.info('PROCESS COMPLETE')
        return X

    def replace_words(self, text: str) -> str:
        for key, value in word_mapper.items():
            text = re.sub(r'\b' + re.escape(key) + r'\b', value, text)
        return text


class TextNoiseRemover(BaseEstimator, TransformerMixin):
    def __init__(self, column_name: str):
        self.__digit_pattern = r'\d'
        self.__new_line = r'\n'
        self.__multiple_spaces = r'\s+'
        self.__column_name = column_name

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        logger.info('NOW PERFORMING PATTERN CLEANING')
        X[self.__column_name] = X[self.__column_name].astype(str).apply(self.remove_patterns)

        logger.info('PROCESS COMPLETE')
        return X

    def remove_patterns(self, text: str) -> str:
        text = re.sub(self.__digit_pattern, '', text)
        text = re.sub(self.__new_line, ' ', text)
        text = re.sub(self.__multiple_spaces, ' ', text)

        return text


class TextNormalizer(BaseEstimator, TransformerMixin):
    def __init__(self, column_name: str, nlp: Language):
        self.__column_name = column_name
        self.__nlp = nlp

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        logger.info('NOW PERFORMING SPACY NORMALIZATION')
        X[self.__column_name] = X[self.__column_name].astype(str).apply(self.preprocess_text)

        logger.info('PROCESS COMPLETE')
        return X

    def preprocess_text(self, text: str) -> str:
        doc = self.__nlp(text)
        return " ".join([token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct])


class DropNullRows(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        logger.info('NOW DROPPING ROWS CONTAINING NULL VALUES')
        return X.dropna()
