import pandas as pd

from sklearn.preprocessing import StandardScaler
from utils import load_df_from_zip


class DataPreprocessor:
    def __init__(self, path_to_zip: str):
        self.path_to_df = path_to_zip
        self.df = load_df_from_zip(self.path_to_df)
        self.scaler = StandardScaler()

    def preprocess_data(self) -> pd.DataFrame:
        self.df = self._drop_columns()
        self.df = self._one_hot_encode()
        self.df = self._replace_values()
        # self.df = self._normalize_values()

        return self.df

    def _one_hot_encode(self) -> pd.DataFrame:
        categorical_columns = ['Vehicle_Age', 'Vehicle_Damage']

        return pd.get_dummies(self.df, columns=categorical_columns)

    def _replace_values(self) -> pd.DataFrame:
        self.df['Gender'] = self.df['Gender'].replace({'Male': 0, 'Female': 1})
        self.df = self.df.replace({col: {True: 1, False: 0} for col in self.df.columns})

        return self.df

    def _drop_columns(self) -> pd.DataFrame:
        return self.df.drop(columns=['id', 'Driving_License'], axis=1)

    def _normalize_values(self) -> pd.DataFrame:
        columns = ['Annual_Premium', 'Vintage', 'Policy_Sales_Channel', 'Age']
        self.df[columns] = self.scaler.fit_transform(self.df[columns])
        return self.df
