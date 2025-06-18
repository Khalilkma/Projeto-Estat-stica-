import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


df = pd.read_csv('../arquivos_csv/obesidade_estados.csv')
print(df.head())

# Medidas de tendência central
media = df['Obesity'].mean()
mediana = df['Obesity'].median()
moda = df['Obesity'].mode()[0]

print(f"Média de Obesidade: {media:.2f}%")
print(f"Mediana: {mediana}%")
print(f"Moda: {moda}%")

# Cálculos de dispersão
desvio_padrao = df['Obesity'].std()
variancia = df['Obesity'].var()

print(f"Desvio Padrão: {desvio_padrao:.2f}")
print(f"Variância: {variancia:.2f}")

# Histograma (Fazer)