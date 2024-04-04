import pandas as pd


class DataPreprocessingPipeline:
    def __init__(self, columns_to_drop: list, categorical_columns: list):
        self.drop_columns = columns_to_drop
        self.categorical_columns = categorical_columns

    def preprocess_data(self, dataset: pd.DataFrame) -> pd.DataFrame:
        dataset = self._drop_columns(dataset)
        dataset = self._one_hot_encoding(dataset)
        dataset = self._handle_bank_balance(dataset)
        return dataset

    def _one_hot_encoding(self, dataset: pd.DataFrame) -> pd.DataFrame:
        dataset = pd.get_dummies(dataset, columns=self.categorical_columns)
        dataset.replace({True: 1, False: 0}, inplace=True)
        return dataset

    def _drop_columns(self, dataset: pd.DataFrame) -> pd.DataFrame:
        dataset.drop(columns=self.drop_columns, inplace=True)
        return dataset

    def _handle_bank_balance(self, dataset: pd.DataFrame) -> pd.DataFrame:
        dataset['Balance'] = dataset['Balance'].apply(lambda x: 1 if x != 0 else 0)
        return dataset
