import pandas as pd

hypertension_df = pd.read_csv('../Data/hypertension_all_2016.csv')
diabetes_df = pd.read_csv('../Data/diabetes_all_2016.csv')

print(hypertension_df.head())
print(diabetes_df.head())