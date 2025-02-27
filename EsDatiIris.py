import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# dataset Iris
dati_iris = load_iris()
df_iris = pd.DataFrame(dati_iris.data, columns = dati_iris.feature_names)
df_iris['specie'] = pd.Categorical.from_codes(dati_iris.target, dati_iris.target_names)

# info base
print("Informazioni sul dataset:")
print(df_iris.info())
print("\nValori nulli presenti:")
print(df_iris.isnull().sum())

# Statistica descrittiva
statistiche_descrittive = df_iris.describe()
print("\nStatistiche descrittive:")
print(statistiche_descrittive)

# deviazione standard per specie
deviazione_standard = df_iris.groupby('specie').std()
print("\nDeviazione standard per specie:")
print(deviazione_standard)

# media per specie
media_specie = df_iris.groupby('specie').mean()
print("\nMedia per specie:")
print(media_specie)

# valore minimo per specie
minimo_specie = df_iris.groupby('specie').min()
print("\nValore minimo per specie:")
print(minimo_specie)

# valore massimo per specie
massimo_specie = df_iris.groupby('specie').max()
print("\nValore massimo per specie:")
print(massimo_specie)
