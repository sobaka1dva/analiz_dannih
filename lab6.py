#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error

# ---------------------------------------------------------
# Настройки
# ---------------------------------------------------------

CSV_PATH = "boston.csv"
OUT_DIR = "lab06_results"
os.makedirs(OUT_DIR, exist_ok=True)
RANDOM_STATE = 42
TARGET = "MEDV"

# ---------------------------------------------------------
# 1. Загрузка данных
# ---------------------------------------------------------

df = pd.read_csv(CSV_PATH)
print("\nИсходные данные:", df.shape)
print(df.head())

# ---------------------------------------------------------
# 2. Приведение типов и заполнение пропусков
# ---------------------------------------------------------

for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

missing_before = df.isnull().sum()
print("\nПропуски до заполнения:\n", missing_before[missing_before > 0])

df = df.fillna(df.median())

missing_after = df.isnull().sum()
print("\nПропуски после заполнения:\n", missing_after[missing_after > 0])

# ---------------------------------------------------------
# 3. Корреляция и heatmap
# ---------------------------------------------------------

corr = df.corr()

plt.figure(figsize=(10, 8))
plt.imshow(corr, cmap="coolwarm", aspect="auto")
plt.colorbar()
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.title("Correlation heatmap")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "heatmap.png"), dpi=150)
plt.close()

# ---------------------------------------------------------
# 4. Выбор 4–6 признаков с наибольшей корреляцией
# ---------------------------------------------------------

corr_with_target = corr[TARGET].abs().sort_values(ascending=False)
top_features = corr_with_target.drop(TARGET).head(5).index.tolist()

print("\nТоп признаков:", top_features)

# ---------------------------------------------------------
# 5. Scatter-plots для выбранных признаков
# ---------------------------------------------------------

for feat in top_features:
    plt.figure(figsize=(6, 4))
    plt.scatter(df[feat], df[TARGET], alpha=0.6)
    plt.xlabel(feat)
    plt.ylabel(TARGET)
    plt.title(f"{feat} vs {TARGET}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"scatter_{feat}.png"), dpi=150)
    plt.close()

# ---------------------------------------------------------
# 6. Формирование выборок
# ---------------------------------------------------------

X = df[top_features]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True, random_state=RANDOM_STATE
)

# ---------------------------------------------------------
# 7. Linear Regression
# ---------------------------------------------------------

lr = LinearRegression()
lr.fit(X_train, y_train)

y_train_pred = lr.predict(X_train)
y_test_pred = lr.predict(X_test)

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

print("\n=== Linear Regression ===")
print("Train R2:", r2_score(y_train, y_train_pred))
print("Test R2:", r2_score(y_test, y_test_pred))
print("Train RMSE:", rmse(y_train, y_train_pred))
print("Test RMSE:", rmse(y_test, y_test_pred))

# ---------------------------------------------------------
# 8. Boxplot + определение выбросов
# ---------------------------------------------------------

plt.figure(figsize=(6, 4))
plt.boxplot(df[TARGET], vert=False)
plt.title("Boxplot MEDV")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "boxplot_MEDV.png"), dpi=150)
plt.close()

Q1 = df[TARGET].quantile(0.25)
Q3 = df[TARGET].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df[TARGET] < lower_bound) | (df[TARGET] > upper_bound)]
print("\nКоличество выбросов:", len(outliers))

# ---------------------------------------------------------
# 9. Повторное обучение без выбросов
# ---------------------------------------------------------

df_no_out = df[(df[TARGET] >= lower_bound) & (df[TARGET] <= upper_bound)]

X_no = df_no_out[top_features]
y_no = df_no_out[TARGET]

Xtr_no, Xte_no, ytr_no, yte_no = train_test_split(
    X_no, y_no, test_size=0.2, random_state=RANDOM_STATE
)

lr_no = LinearRegression()
lr_no.fit(Xtr_no, ytr_no)

print("\n=== Linear Regression (без выбросов) ===")
print("Train R2:", r2_score(ytr_no, lr_no.predict(Xtr_no)))
print("Test R2:", r2_score(yte_no, lr_no.predict(Xte_no)))
print("Train RMSE:", rmse(ytr_no, lr_no.predict(Xtr_no)))
print("Test RMSE:", rmse(yte_no, lr_no.predict(Xte_no)))

# ---------------------------------------------------------
# 10. Ridge Regression
# ---------------------------------------------------------

ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)

print("\n=== Ridge Regression ===")
print("Train R2:", r2_score(y_train, ridge.predict(X_train)))
print("Test R2:", r2_score(y_test, ridge.predict(X_test)))
print("Train RMSE:", rmse(y_train, ridge.predict(X_train)))
print("Test RMSE:", rmse(y_test, ridge.predict(X_test)))

# ---------------------------------------------------------
# 11. Polynomial Regression (degree=3)
# ---------------------------------------------------------

poly_model = Pipeline([
    ("poly", PolynomialFeatures(degree=3, include_bias=False)),
    ("scaler", StandardScaler()),
    ("lr", LinearRegression())
])

poly_model.fit(X_train, y_train)

print("\n=== Polynomial Regression (deg=3) ===")
print("Train R2:", r2_score(y_train, poly_model.predict(X_train)))
print("Test R2:", r2_score(y_test, poly_model.predict(X_test)))
print("Train RMSE:", rmse(y_train, poly_model.predict(X_train)))
print("Test RMSE:", rmse(y_test, poly_model.predict(X_test)))

print("\nГотово! Все графики сохранены в папку:", OUT_DIR)
