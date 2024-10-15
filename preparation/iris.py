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
