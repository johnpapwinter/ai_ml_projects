import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


class DataPreprocessor:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.scaler = MinMaxScaler()
        self.one_hot_encoder = OneHotEncoder(handle_unknown='ignore')
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('onehot', self.one_hot_encoder,
                 ['Gender', 'Vehicle_Age', 'Vehicle_Damage', 'Region_Code', 'Policy_Sales_Channel']),
                ('scaler', self.scaler, ['Age', 'Annual_Premium', 'Vintage'])
            ],
            remainder='passthrough'
        )

    def preprocess_data(self, test_data: pd.DataFrame = None) -> pd.DataFrame:
        if not test_data:
            return self.preprocessor.fit_transform(self.df)
        else:
            return self.preprocessor.transform(test_data)

    def get_feature_names_out(self):
        return self.preprocessor.get_feature_names_out()

