import pandas as pd
import logging

from keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input, Dense, BatchNormalization
from tensorflow.keras.models import Sequential

import logging_config

logging_config.configure_logging()
LOGGER = logging.getLogger(__name__)


class BankChurnClassifier:
    def __init__(self):
        self.input_features = 13
        self.model = None
        self.build_model()

    def build_model(self):
        self.model = Sequential([
            Input(shape=(self.input_features,), name='INPUT'),
            Dense(128, activation='relu', name='L1'),
            BatchNormalization(),
            Dense(64, activation='relu', name='L2'),
            BatchNormalization(),
            Dense(32, activation='relu', name='L3'),
            BatchNormalization(),
            Dense(1, activation='sigmoid', name='OUT')
        ])
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    def train_model(self,
                    X_train: pd.DataFrame,
                    y_train: pd.DataFrame,
                    epochs: int,
                    batch_size: int,
                    validation_split: float):
        early_stopping = EarlyStopping(patience=10, min_delta=0.001, monitor='val_loss', restore_best_weights=True)

        LOGGER.info(f"Starting model training")
        history = self.model.fit(X_train,
                                 y_train,
                                 epochs=epochs,
                                 batch_size=batch_size,
                                 validation_split=validation_split,
                                 callbacks=[early_stopping]
                                 )
        LOGGER.info(f"Finished model training")

    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.DataFrame) -> tuple:
        loss, accuracy = self.model.evaluate(X_test, y_test)

        LOGGER.info(f"Loss of model: {loss}")
        LOGGER.info(f"Accuracy of model: {accuracy}")
        return loss, accuracy

    def get_predictions(self, X_values: pd.DataFrame) -> pd.DataFrame:
        predicted_classes = (self.model.predict(X_values) > 0.5).astype(int)
        return predicted_classes[:, 0]

    def save_model(self):
        self.model.save('bank_churn_weights.h5')
        LOGGER.info("Model saved")

