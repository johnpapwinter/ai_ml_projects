import pandas as pd

from tensorflow.keras.layers import InputLayer, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping


class CrossSellingModel:
    def __init__(self, input_shape: int):
        self.model = Sequential()
        self._create_model(input_shape)

    def get_model(self) -> Sequential:
        return self.model

    def save_model(self, model_name: str) -> None:
        self.model.save(model_name)

    def _create_model(self, input_shape: int) -> None:
        self.model.add(InputLayer(shape=(input_shape,)))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))

        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    def train_model(self,
                    X_train: pd.DataFrame,
                    y_train: pd.DataFrame,
                    epochs: int,
                    batch_size: int,
                    validation_split: float):
        early_stopping = EarlyStopping(patience=10, min_delta=0.001, monitor='val_loss', restore_best_weights=True)

        history = self.model.fit(X_train,
                                 y_train,
                                 epochs=epochs,
                                 batch_size=batch_size,
                                 validation_split=validation_split,
                                 callbacks=[early_stopping]
                                 )

    def get_predictions(self, X_values: pd.DataFrame, threshold: float) -> pd.DataFrame:
        predictions = (self.model.predict(X_values) > threshold).astype(int)
        return predictions[:, 0]
