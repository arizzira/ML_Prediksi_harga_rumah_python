import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# === Data Dummy ===
data = {
    'luas': [60, 80, 100, 120, 150],
    'kamar': [2, 3, 3, 4, 4],
    'harga': [300, 400, 500, 600, 750]
}

df = pd.DataFrame(data)
print("==== Data ====")
print(df)

# === Pisahkan input dan output ===
X = df[['luas', 'kamar']]   # fitur
y = df['harga']             # target

# === Split data ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Model Linear Regression ===
model = LinearRegression()
model.fit(X_train, y_train)

# === Prediksi ===
y_pred = model.predict(X_test)
print("\n=== Prediksi ===")
for pred, asli in zip(y_pred, y_test):
    print(f"Prediksi: {pred:.2f} juta, Asli: {asli} juta")

# === Evaluasi Model ===
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("\n=== Evaluasi Model ===")
print("MAE  :", round(mae, 2))
print("MSE  :", round(mse, 2))
print("RMSE :", round(rmse, 2))

# === Visualisasi ===
plt.scatter(y_test, y_pred, color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.xlabel("Harga Asli (juta)")
plt.ylabel("Harga Prediksi (juta)")
plt.title("Prediksi vs Harga Asli")
plt.grid(True)
plt.show()

# === Prediksi Rumah Baru ===
rumah_baru = np.array([[90, 3]])
prediksi = model.predict(rumah_baru)
print(f"\nPrediksi harga rumah 90m² dengan 3 kamar: {prediksi[0]:.2f} juta")