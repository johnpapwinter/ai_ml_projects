import pandas as pd

from tensorflow.keras.layers import InputLayer, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping


class CrossSellingModel:
    def __init__(self, input_shape: int):
        self.model = Sequential()
        self._create_model(input_shape)

    def get_model(self) -> Sequential:
        return self.model

    def save_model(self, model_name: str) -> None:
        self.model.save(f"{model_name}.keras")

    def _create_model(self, input_shape: int) -> None:
        self.model.add(InputLayer(shape=(input_shape,)))
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(16, activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))

        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['AUC'])

    def train_model(self,
                    X_train: pd.DataFrame,
                    y_train: pd.DataFrame,
                    X_test: pd.DataFrame,
                    y_test: pd.DataFrame,
                    epochs: int,
                    batch_size: int):
        early_stopping = EarlyStopping(patience=5, min_delta=0.001, monitor='val_loss', restore_best_weights=True)

        self.model.fit(X_train,
                       y_train,
                       epochs=epochs,
                       batch_size=batch_size,
                       validation_data=(X_test, y_test),
                       callbacks=[early_stopping],
                       verbose=2
                       )

    def get_predictions(self, X_values: pd.DataFrame, threshold: float) -> pd.DataFrame:
        predictions = (self.model.predict(X_values) > threshold).astype(int)
        return predictions[:, 0]

    def prepare_kaggle_submission(self, df: pd.DataFrame, ids: pd.DataFrame) -> None:
        predictions = self.get_predictions(df, 0.5)
        submission = pd.DataFrame({'id': ids, 'Response': predictions})
        submission.to_csv('submission.csv', index=False)


