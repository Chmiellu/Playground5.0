#%%
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Wczytanie danych
telco_data = pd.read_csv("Telco-Customer-Churn.csv")
data_dumies = pd.get_dummies(telco_data, drop_first=False)

# 1.1. Sprawdzenie typów danych i podstawowe statystyki
telco_data['TotalCharges'] = pd.to_numeric(telco_data['TotalCharges'], errors='coerce')
telco_data = telco_data.dropna()  # Usunięcie brakujących wartości
print(telco_data.info())
print(telco_data.describe(include="all"))

# 1.2. Utworzenie wykresów słupkowych
# 1.2.1. Ilości wystąpień kategorii w zmiennej decyzyjnej
sns.countplot(data=telco_data, x="Churn", palette="pastel")
plt.title("Rozkład zmiennej 'Churn'")
plt.xlabel("Churn")
plt.ylabel("Liczba klientów")
plt.show()

# 1.2.2. Zmienna decyzyjna według płci
sns.countplot(data=telco_data, x="Churn", hue="gender", palette="pastel")
plt.title("Rozkład 'Churn' według płci")
plt.xlabel("Churn")
plt.ylabel("Liczba klientów")
plt.legend(title="Płeć")
plt.show()

# 1.3. Wykresy pudełkowe
# 1.3.1. Rozkład całkowitych opłat (TotalCharges) względem 'Churn'
sns.boxplot(data=telco_data, x="Churn", y="TotalCharges", palette="pastel")
plt.title("Opłaty całkowite względem 'Churn'")
plt.xlabel("Churn")
plt.ylabel("Całkowite opłaty")
plt.show()

# 1.3.2. Rozkład 'Churn' względem długości korzystania z usług (tenure)
sns.boxplot(data=telco_data, x="Churn", y="tenure", palette="pastel")
plt.title("Długość korzystania z usług względem 'Churn'")
plt.xlabel("Churn")
plt.ylabel("Długość korzystania z usług (miesiące)")
plt.show()

# 1.4. Opłaty całkowite względem rodzaju umowy i 'Churn'
sns.boxplot(data=telco_data, x="Contract", y="TotalCharges", hue="Churn", palette="pastel")
plt.title("Opłaty całkowite względem rodzaju umowy i 'Churn'")
plt.xlabel("Rodzaj umowy")
plt.ylabel("Całkowite opłaty")
plt.legend(title="Churn")
plt.show()

# 1.5. Korelacje cech z 'Churn'

from sklearn.preprocessing import LabelEncoder

telco_data_encoded = telco_data.drop(columns=["customerID", "tenure", "MonthlyCharges", "TotalCharges"])

for col in telco_data_encoded.select_dtypes(include='object').columns:
    telco_data_encoded[col] = LabelEncoder().fit_transform(telco_data_encoded[col])

correlation = telco_data_encoded.corr()["Churn"].sort_values(ascending=False).drop("Churn")

plt.figure(figsize=(10, 6))
correlation.plot(kind="bar", color="skyblue")
plt.title("Korelacje cech z 'Churn'")
plt.ylabel("Współczynnik korelacji")
plt.show()

# 1.6. Rodzaje zawieranych umów
print("Rodzaje umów:", telco_data["Contract"].unique())

# 1.7. Rozkład różnych usług używanych przez klientów
services = ["PhoneService", "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
for service in services:
    sns.countplot(data=telco_data, x=service, hue="Churn", palette="pastel")
    plt.title(f"Rozkład usługi {service} względem 'Churn'")
    plt.xlabel(service)
    plt.ylabel("Liczba klientów")
    plt.legend(title="Churn")
    plt.show()

# 1.8. Histogram długości korzystania z usług (tenure)
sns.histplot(data=telco_data, x="tenure", bins=30, kde=True, color="blue")
plt.title("Rozkład długości korzystania z usług (tenure)")
plt.xlabel("Długość korzystania z usług (miesiące)")
plt.ylabel("Liczba klientów")
plt.show()

# 1.9. Histogram długości korzystania z usług według 'Churn' i typu umowy
sns.histplot(data=telco_data, x="tenure", bins=30, kde=True, hue="Contract", multiple="stack", palette="pastel")
plt.title("Rozkład długości korzystania z usług według 'Churn' i typu umowy")
plt.xlabel("Długość korzystania z usług (miesiące)")
plt.ylabel("Liczba klientów")
plt.show()

# 1.10. Współczynnik rezygnacji dla każdej kohorty (tenure)
# Upewnij się, że kolumna 'Churn' jest numeryczna
telco_data['Churn'] = telco_data['Churn'].map({'Yes': 1, 'No': 0})

# Obliczenie współczynnika rezygnacji (średnia wartości 'Churn' w grupach 'tenure')
churn_rates = telco_data.groupby("tenure")["Churn"].mean()

# Wykres współczynnika rezygnacji
plt.plot(churn_rates.index, churn_rates.values, marker="o", color="blue")
plt.title("Współczynnik rezygnacji w zależności od 'tenure'")
plt.xlabel("Długość korzystania z usług (miesiące)")
plt.ylabel("Współczynnik rezygnacji")
plt.show()


# 1.11. Zależność współczynnika rezygnacji od 'tenure'
# (Realizowane w punkcie 1.10, ponieważ wykres jest już opisany jako zależność).

# 1.12. Podział 'tenure' na grupy
bins = [0, 12, 24, 48, np.inf]
labels = ['0-12 miesięcy', '12-24 miesiące', '24-48 miesięcy', 'Powyżej 48 miesięcy']
telco_data["tenure_group"] = pd.cut(telco_data["tenure"], bins=bins, labels=labels)

# 1.13. Wykres pudełkowy dla grup 'tenure' względem 'Churn'
sns.boxplot(data=telco_data, x="tenure_group", y="tenure", hue="Churn", palette="pastel")
plt.title("Długość korzystania z usług podzielona na grupy względem 'Churn'")
plt.xlabel("Grupa 'tenure'")
plt.ylabel("Długość korzystania z usług (miesiące)")
plt.legend(title="Churn")
plt.show()

#%%

#%%
from sklearn.model_selection import train_test_split

# 2.1. Podział danych na macierz cech (X) i etykiety (y)
X = telco_data_encoded.drop(columns=["Churn"])
y = telco_data_encoded["Churn"]

# 2.2. Podział na zbiór treningowy i testowy z zachowaniem random_state=42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1100, random_state=42)

print("Rozmiar zbioru treningowego (X_train):", X_train.shape)
print("Rozmiar zbioru testowego (X_test):", X_test.shape)
print("Rozmiar etykiet treningowych (y_train):", y_train.shape)
print("Rozmiar etykiet testowych (y_test):", y_test.shape)

#%%

#%%
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# 3.1. Utworzenie modelu drzewa decyzyjnego z domyślnymi parametrami
tree_clf = DecisionTreeClassifier(random_state=42)
tree_clf.fit(X_train, y_train)

# 3.2. Predykcja na zbiorze testowym
y_pred_tree = tree_clf.predict(X_test)

# Wyświetlenie raportu klasyfikacji
print("Raport klasyfikacji dla drzewa decyzyjnego:")
print(classification_report(y_test, y_pred_tree))

# Wyświetlenie macierzy błędów
cm = confusion_matrix(y_test, y_pred_tree, labels=tree_clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=tree_clf.classes_)
disp.plot(cmap="Blues")
plt.xlabel("Przewidywana etykieta klasy")
plt.ylabel("Rzeczywista etykieta klasy")
plt.title("Macierz błędów - Drzewo decyzyjne")
plt.show()

# 3.3. Wizualizacja drzewa decyzyjnego
plt.figure(figsize=(20, 10))
plot_tree(tree_clf, feature_names=X.columns, class_names=["No", "Yes"], filled=True)
plt.title("Drzewo decyzyjne")
plt.show()

# Wyświetlenie ważności cech
feature_importances = pd.Series(tree_clf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("Ważność cech:")
print(feature_importances)

#%%
from sklearn.neighbors import KNeighborsClassifier

# 4.1. Utworzenie modelu K-Najbliższych Sąsiadów z domyślnymi parametrami
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_train)

# 4.2. Predykcja na zbiorze testowym
y_pred_knn = knn_clf.predict(X_test)

print("Raport klasyfikacji dla K-Najbliższych Sąsiadów:")
print(classification_report(y_test, y_pred_knn))

# Wyświetlenie macierzy błędów
cm_knn = confusion_matrix(y_test, y_pred_knn, labels=knn_clf.classes_)
disp_knn = ConfusionMatrixDisplay(confusion_matrix=cm_knn, display_labels=knn_clf.classes_)
disp_knn.plot(cmap="Blues")
plt.xlabel("Przewidywana etykieta klasy")
plt.ylabel("Rzeczywista etykieta klasy")
plt.title("Macierz błędów - K-Najbliżsi Sąsiedzi")
plt.show()

#%%

#%%
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

logreg_default = LogisticRegression(random_state=42, max_iter=1000)
logreg_default.fit(X_train, y_train)

y_pred_logreg = logreg_default.predict(X_test)

# Generowanie raportu z klasyfikacji
print("Raport klasyfikacji - Regresja Logistyczna (ustawienia domyślne):")
print(classification_report(y_test, y_pred_logreg, target_names=["No Churn", "Churn"]))

# Rysowanie macierzy błędów
ConfusionMatrixDisplay.from_estimator(
    logreg_default, X_test, y_test, display_labels=["No Churn", "Churn"], cmap="Blues"
)
plt.title("Macierz błędów - Regresja Logistyczna (ustawienia domyślne)")
plt.xlabel("Przewidywana etykieta klasy")
plt.ylabel("Rzeczywista etykieta klasy")
plt.show()

