import pandas as pd

df = pd.read_csv('submission.csv')
print(df.head())
print("*" * 30)

print(df['Response'].value_counts())
