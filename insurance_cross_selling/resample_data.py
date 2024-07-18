import pandas as pd
import logging
from sklearn.utils import resample

import logging_config

logging_config.configure_logging()
LOGGER = logging.getLogger(__name__)


def resample_data(df: pd.DataFrame) -> pd.DataFrame:
    # features = df.drop('Response', axis=1)
    target = df['Response']

    majority_class_samples = 1415000
    undersampled_majority = resample(df[target == 0], replace=False, n_samples=majority_class_samples, random_state=42)
    LOGGER.info(f"Undersampled: {df.shape}")

    return pd.concat([undersampled_majority, df[target == 1]])
