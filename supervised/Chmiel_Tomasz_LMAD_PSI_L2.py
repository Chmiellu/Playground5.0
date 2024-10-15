import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv('data/housing.csv', delim_whitespace=True, header=None)

# Nadanie nazw kolumnom
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

# Sprawdzenie, czy dane zostały poprawnie wczytane
print(df.head())
print(df.shape)  # Powinno pokazać 506 wierszy i 14 kolumn

# Sprawdzenie podstawowych statystyk
print(df.describe())
print(df.isnull().sum())

print(df.dtypes)

df.boxplot(figsize=(12, 8))
plt.xticks(rotation=90)
plt.show()

outliers_percentage = {}
for column in df.columns:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    outliers_percentage[column] = 100 * len(outliers) / len(df)
print(outliers_percentage)

corr_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()

sns.pairplot(df)
plt.show()

df_selected = df[df.columns[df.corr()['MEDV'].abs() > 0.5]]
sns.pairplot(df_selected)
plt.show()

X = df.drop('MEDV', axis=1)

y = df['MEDV']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

model = LinearRegression()
model.fit(X_train, y_train)

print("Współczynniki:", model.coef_)
print("Punkt przecięcia (intercept):", model.intercept_)

y_pred = model.predict(X_test)

plt.scatter(y_test, y_pred)
plt.xlabel('Rzeczywiste wartości')
plt.ylabel('Przewidywane wartości')
plt.show()

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print(f"MAE: {mae}, MSE: {mse}, RMSE: {rmse}, R2: {r2}")

ridge = Ridge(alpha=0.5)
ridge.fit(X_train, y_train)
ridge_pred = ridge.predict(X_test)
ridge_mae = mean_absolute_error(y_test, ridge_pred)
ridge_r2 = r2_score(y_test, ridge_pred)
print(f"Ridge MAE: {ridge_mae}, R2: {ridge_r2}")

lasso = Lasso(alpha=0.5)
lasso.fit(X_train, y_train)
lasso_pred = lasso.predict(X_test)
lasso_mae = mean_absolute_error(y_test, lasso_pred)
lasso_r2 = r2_score(y_test, lasso_pred)
print(f"Lasso MAE: {lasso_mae}, R2: {lasso_r2}")

elastic = ElasticNet(alpha=0.5, l1_ratio=0.5)
elastic.fit(X_train, y_train)
elastic_pred = elastic.predict(X_test)
elastic_mae = mean_absolute_error(y_test, elastic_pred)
elastic_r2 = r2_score(y_test, elastic_pred)
print(f"ElasticNet MAE: {elastic_mae}, R2: {elastic_r2}")



X_selected = df[['RM', 'PTRATIO', 'LSTAT']]
y = df['MEDV']

X_train_sel, X_test_sel, y_train_sel, y_test_sel = train_test_split(X_selected, y, test_size=0.2, random_state=101)

model_sel = LinearRegression()
model_sel.fit(X_train_sel, y_train_sel)

print("Współczynniki dla zredukowanego modelu:", model_sel.coef_)
print("Punkt przecięcia (intercept) dla zredukowanego modelu:", model_sel.intercept_)

y_pred_sel = model_sel.predict(X_test_sel)

plt.scatter(y_test_sel, y_pred_sel)
plt.xlabel('Rzeczywiste wartości')
plt.ylabel('Przewidywane wartości')
plt.title('Rzeczywiste vs Przewidywane (Zredukowane cechy)')
plt.show()

mae_sel = mean_absolute_error(y_test_sel, y_pred_sel)
mse_sel = mean_squared_error(y_test_sel, y_pred_sel)
rmse_sel = np.sqrt(mse_sel)
r2_sel = r2_score(y_test_sel, y_pred_sel)

print(f"MAE (po redukcji): {mae_sel}, MSE: {mse_sel}, RMSE: {rmse_sel}, R2: {r2_sel}")

ridge_sel = Ridge(alpha=0.5)
ridge_sel.fit(X_train_sel, y_train_sel)
ridge_pred_sel = ridge_sel.predict(X_test_sel)
ridge_mae_sel = mean_absolute_error(y_test_sel, ridge_pred_sel)
ridge_r2_sel = r2_score(y_test_sel, ridge_pred_sel)
print(f"Ridge MAE (po redukcji): {ridge_mae_sel}, R2: {ridge_r2_sel}")

lasso_sel = Lasso(alpha=0.5)
lasso_sel.fit(X_train_sel, y_train_sel)
lasso_pred_sel = lasso_sel.predict(X_test_sel)
lasso_mae_sel = mean_absolute_error(y_test_sel, lasso_pred_sel)
lasso_r2_sel = r2_score(y_test_sel, lasso_pred_sel)
print(f"Lasso MAE (po redukcji): {lasso_mae_sel}, R2: {lasso_r2_sel}")

elastic_sel = ElasticNet(alpha=0.5, l1_ratio=0.5)
elastic_sel.fit(X_train_sel, y_train_sel)
elastic_pred_sel = elastic_sel.predict(X_test_sel)
elastic_mae_sel = mean_absolute_error(y_test_sel, elastic_pred_sel)
elastic_r2_sel = r2_score(y_test_sel, elastic_pred_sel)
print(f"ElasticNet MAE (po redukcji): {elastic_mae_sel}, R2: {elastic_r2_sel}")




import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv('data/housing.csv', delim_whitespace=True, header=None)
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

X = df.drop('MEDV', axis=1)
y = df['MEDV']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

scaler = StandardScaler()

scaler.fit(X_train)

scaled_X_train = scaler.transform(X_train)

scaled_X_test = scaler.transform(X_test)

std_model = LinearRegression()


std_model.fit(scaled_X_train, y_train)

print("Współczynniki dopasowania: ", std_model.coef_)
print("Przecięcie: ", std_model.intercept_)

y_pred = std_model.predict(scaled_X_test)

plt.scatter(y_test, y_pred)
plt.xlabel("Rzeczywiste wartości MEDV")
plt.ylabel("Przewidywane wartości MEDV")
plt.title("Wykres punktowy: rzeczywiste vs przewidywane wartości")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')
plt.show()

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"R2: {r2}")

ridge = Ridge(alpha=0.5)
ridge.fit(scaled_X_train, y_train)
ridge_pred = ridge.predict(scaled_X_test)

print("Ridge - współczynniki dopasowania: ", ridge.coef_)
print(f"Ridge MAE: {mean_absolute_error(y_test, ridge_pred)}")
print(f"Ridge MSE: {mean_squared_error(y_test, ridge_pred)}")
print(f"Ridge RMSE: {np.sqrt(mean_squared_error(y_test, ridge_pred))}")
print(f"Ridge R2: {r2_score(y_test, ridge_pred)}")

lasso = Lasso(alpha=0.5)
lasso.fit(scaled_X_train, y_train)
lasso_pred = lasso.predict(scaled_X_test)

print("Lasso - współczynniki dopasowania: ", lasso.coef_)
print(f"Lasso MAE: {mean_absolute_error(y_test, lasso_pred)}")
print(f"Lasso MSE: {mean_squared_error(y_test, lasso_pred)}")
print(f"Lasso RMSE: {np.sqrt(mean_squared_error(y_test, lasso_pred))}")
print(f"Lasso R2: {r2_score(y_test, lasso_pred)}")

elastic = ElasticNet(alpha=0.5, l1_ratio=0.5)
elastic.fit(scaled_X_train, y_train)
elastic_pred = elastic.predict(scaled_X_test)

print("ElasticNet - współczynniki dopasowania: ", elastic.coef_)
print(f"ElasticNet MAE: {mean_absolute_error(y_test, elastic_pred)}")
print(f"ElasticNet MSE: {mean_squared_error(y_test, elastic_pred)}")
print(f"ElasticNet RMSE: {np.sqrt(mean_squared_error(y_test, elastic_pred))}")
print(f"ElasticNet R2: {r2_score(y_test, elastic_pred)}")


X_selected = df[['RM', 'PTRATIO', 'LSTAT']]

X_train_sel, X_test_sel, y_train_sel, y_test_sel = train_test_split(X_selected, y, test_size=0.2, random_state=101)

scaler_sel = StandardScaler()
scaler_sel.fit(X_train_sel)
scaled_X_train_sel = scaler_sel.transform(X_train_sel)
scaled_X_test_sel = scaler_sel.transform(X_test_sel)

std_model_sel = LinearRegression()
std_model_sel.fit(scaled_X_train_sel, y_train_sel)

ridge_sel = Ridge(alpha=0.5)
ridge_sel.fit(scaled_X_train_sel, y_train_sel)

lasso_sel = Lasso(alpha=0.5)
lasso_sel.fit(scaled_X_train_sel, y_train_sel)

elastic_sel = ElasticNet(alpha=0.5, l1_ratio=0.5)
elastic_sel.fit(scaled_X_train_sel, y_train_sel)


