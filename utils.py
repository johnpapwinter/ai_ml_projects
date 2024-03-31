import os
import zipfile
import logging
import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.base import ClassifierMixin
from sklearn.base import RegressorMixin
from typing import Union
from datetime import datetime

import logging_config

logging_config.configure_logging()
LOGGER = logging.getLogger(__name__)


def load_df_from_zip(zip_filepath: str) -> pd.DataFrame:
    with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
        csv_filename = zip_ref.namelist()[0]
        with zip_ref.open(csv_filename) as file:
            df = pd.read_csv(file)
            return df


def save_df_to_zip(df: pd.DataFrame, filename: str) -> None:
    day = datetime.now().day
    month = datetime.now().month
    year = datetime.now().year
    csv_filename = f'{filename}_{year}_{month}_{day}.csv'
    zip_filename = f'{filename}_{year}_{month}_{day}.zip'

    with zipfile.ZipFile(zip_filename, 'w', compression=zipfile.ZIP_DEFLATED) as zip_file:
        with zip_file.open(csv_filename, 'w') as file:
            df.to_csv(file, index=False)


def get_zip_if_exists(directory: str) -> str | None:
    files = os.listdir(directory)
    target_files = [filename for filename in files if filename.endswith('.zip')]

    if target_files:
        filename = target_files[0]
        return filename
    else:
        return None


def evaluate_prediction(y_test: np.ndarray, y_pred: np.ndarray) -> None:
    accuracy = accuracy_score(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    LOGGER.info(f"Accuracy of model: {accuracy}")
    LOGGER.info(f"Confusion matrix of model: {matrix}")
    LOGGER.info(f"Classification Report of model: {class_report}")


def evaluate_with_cross_validation(model: Union[ClassifierMixin, RegressorMixin],
                                   X_train: np.ndarray,
                                   y_train: np.ndarray) -> None:
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)

    LOGGER.info(f"Cross-validation scores: {cv_scores}")

    mean_cv_score = cv_scores.mean()
    std_cv_score = cv_scores.std()

    LOGGER.info(f"Mean cross-validation score: {mean_cv_score}")
    LOGGER.info(f"Standard deviation of cross-validation scores: {std_cv_score}")

