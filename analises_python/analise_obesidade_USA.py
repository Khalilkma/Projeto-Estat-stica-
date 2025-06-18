import pandas as pd
import numpy as np
import matplotlib as plt


df = pd.read_csv('../arquivos_csv/obesidade_estados.csv')
print(df.head())

media = df['Obesity'].mean()
mediana = df['Obesity'].median()
moda = df['Obesity'].mode()[0]

print(f"Média de Obesidade: {media:.2f}%")
print(f"Mediana: {mediana}%")
print(f"Moda: {moda}%")
