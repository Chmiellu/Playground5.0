import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns


iris = load_iris(as_frame=True)
X = iris['data']
y = iris['target']

species_map = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
y = y.map(species_map)

df = pd.concat([X, y.rename("species")], axis=1)

print(df.describe(include='all'))

print(df.isnull().sum())

scaler_01 = MinMaxScaler(feature_range=(0, 1))
df_normalized_01 = df.copy()
df_normalized_01.iloc[:, :-1] = scaler_01.fit_transform(df.iloc[:, :-1])

scaler_11 = MinMaxScaler(feature_range=(-1, 1))
df_normalized_11 = df.copy()
df_normalized_11.iloc[:, :-1] = scaler_11.fit_transform(df.iloc[:, :-1])

scaler_std = StandardScaler()
df_standardized = df.copy()
df_standardized.iloc[:, :-1] = scaler_std.fit_transform(df.iloc[:, :-1])

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
sns.scatterplot(data=df, x="petal length (cm)", y="petal width (cm)", hue="species", ax=axes[0, 0])
axes[0, 0].set_title("Original Data")

sns.scatterplot(data=df_normalized_01, x="petal length (cm)", y="petal width (cm)", hue="species", ax=axes[0, 1])
axes[0, 1].set_title("Normalized (0, 1)")

sns.scatterplot(data=df_normalized_11, x="petal length (cm)", y="petal width (cm)", hue="species", ax=axes[1, 0])
axes[1, 0].set_title("Normalized (-1, 1)")

sns.scatterplot(data=df_standardized, x="petal length (cm)", y="petal width (cm)", hue="species", ax=axes[1, 1])
axes[1, 1].set_title("Standardized")

plt.tight_layout()
plt.show()


fig, axes = plt.subplots(2, 2, figsize=(14, 10))
sns.scatterplot(data=df, x="sepal length (cm)", y="sepal width (cm)", hue="species", ax=axes[0, 0])
axes[0, 0].set_title("Original Data")

sns.scatterplot(data=df_normalized_01, x="sepal length (cm)", y="sepal width (cm)", hue="species", ax=axes[0, 1])
axes[0, 1].set_title("Normalized (0, 1)")

sns.scatterplot(data=df_normalized_11, x="sepal length (cm)", y="sepal width (cm)", hue="species", ax=axes[1, 0])
axes[1, 0].set_title("Normalized (-1, 1)")

sns.scatterplot(data=df_standardized, x="sepal length (cm)", y="sepal width (cm)", hue="species", ax=axes[1, 1])
axes[1, 1].set_title("Standardized")

plt.tight_layout()
plt.show()


import pandas as pd
import matplotlib.pyplot as plt


file_path = 'data/Zad2_L1.csv'
df = pd.read_csv(file_path, sep=';', decimal=',', skiprows=[1])

df.iloc[:, 1:] = df.iloc[:, 1:].apply(pd.to_numeric)

wavenumber = df['Wavenumber [cm^-1]']
intensities = df.drop(columns=['Wavenumber [cm^-1]'])

index_985 = (wavenumber - 985).abs().idxmin()

intensities_normalized = intensities / intensities.iloc[index_985]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

ax1.plot(wavenumber, intensities)
ax1.set_title('Raw Raman Spectra')
ax1.set_xlabel('Wavenumber [cm^-1]')
ax1.set_ylabel('Intensity [a.u.]')

ax2.plot(wavenumber, intensities_normalized)
ax2.set_title('Normalized Raman Spectra (normalized to 985 cm^-1)')
ax2.set_xlabel('Wavenumber [cm^-1]')
ax2.set_ylabel('Normalized Intensity [a.u.]')

plt.tight_layout()
plt.show()



import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


file_path = 'data/Zad3_L1.csv'
df = pd.read_csv(file_path, sep=';', decimal=',')

wavenumber = df['Wavenumber [cm^-1]']
absorbance = df.drop(columns=['Wavenumber [cm^-1]'])

area_under_curve = np.trapz(absorbance, wavenumber, axis=0)
absorbance_normalized = absorbance.div(area_under_curve, axis=1)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

for col in absorbance.columns:
    ax1.plot(wavenumber, absorbance[col], label=col)
ax1.set_title('Raw FTIR Spectra')
ax1.set_xlabel('Wavenumber [cm^-1]')
ax1.set_ylabel('Absorbance [a.u.]')
ax1.legend(loc='upper right')

for col in absorbance_normalized.columns:
    ax2.plot(wavenumber, absorbance_normalized[col], label=col)
ax2.set_title('Normalized FTIR Spectra')
ax2.set_xlabel('Wavenumber [cm^-1]')
ax2.set_ylabel('Normalized Absorbance [a.u.]')
ax2.legend(loc='upper right')

plt.tight_layout()
plt.show()


