import spacy
import logging
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

from pipeline_components import *

import logging_config

logging_config.configure_logging()
LOGGER = logging.getLogger(__name__)

TEXT_COLUMN = 'text'
GENDER_COLUMN = 'sex'
nlp = spacy.load('en_core_web_sm')


class GenderClassifierPipeline:
    def __init__(self):
        self.text_column_name = 'text'
        self.gender_column_name = 'sex'
        self.nlp = spacy.load('en_core_web_sm')

    def get_data_pipeline(self) -> Pipeline:
        return Pipeline(steps=[
            ("resampler", DataResampler(self.gender_column_name)),
            ("drop", DropAndTransformHandler(self.text_column_name)),
            ("pattern_cleaner", TextPatternCleaner(self.text_column_name)),
            ("normalizer", TextNormalizer(self.text_column_name, self.nlp)),
            ("drop_null", DropNullRows(self.text_column_name))
        ])

    def get_model_pipeline(self, params: dict | None) -> Pipeline:
        if params is None:
            return Pipeline(steps=[
                ("vectorizer", TfidfVectorizer()),
                ("model", RandomForestClassifier())
            ])
        else:
            return Pipeline(steps=[
                ("vectorizer", TfidfVectorizer()),
                ("model", RandomForestClassifier(**params))
            ])

    def fine_tune_pipeline(self, params: dict, X_train: pd.Series, y_train: pd.Series) -> dict:
        random_search = RandomizedSearchCV(self.get_model_pipeline(params=None),
                                           params,
                                           cv=5,
                                           n_iter=10,
                                           n_jobs=-1)
        random_search.fit(X_train, y_train)

        LOGGER.info(f"Best hyperparameters: {random_search.best_params_}")
        LOGGER.info(f"Best score: {random_search.best_score_}")

        return self.__remove_prefix_from_params(params=random_search.best_params_, prefix="model__")

    def save_model_pipeline(self, model_pipeline: Pipeline, filename: str) -> None:
        try:
            joblib.dump(model_pipeline, filename)
            LOGGER.info(f"MODEL PIPELINE SUCCESSFULLY SAVED TO FILE: {filename}")
        except Exception as e:
            LOGGER.error(f"ERROR OCCURRED WHILE SAVING MODEL PIPELINE: {e}")

    def __remove_prefix_from_params(self, params: dict, prefix: str) -> dict:
        return {key.replace(prefix, ''): value for key, value in params.items()}
