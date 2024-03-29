import zipfile
import pandas as pd

from datetime import datetime


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


