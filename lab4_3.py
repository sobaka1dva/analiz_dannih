"""
ЗАДАНИЕ 3. Анализ клиентской базы (telecom_churn.csv)

1) Пропуски по столбцам:
State                     0
Area code                 0
International plan        0
Number vmail messages     0
Total day minutes         0
Total day calls           0
Total eve minutes         0
Total eve calls           0
Total night minutes       0
Total night calls         0
Customer service calls    0
Churn                     0

2) Статусы клиентов (value_counts) и доли (%):
False    2850
True      483

False    85.51
True     14.49 %

3) Добавлен столбец "Avg call duration" = (дневн.+вечерн.+ночные минуты) / (дневн.+вечерн.+ночные звонки);
   выводятся топ‑10 клиентов с наибольшим значением.
4) Средняя длительность одного звонка по группам Churn:
Churn
False    1.938102
True     2.091193

5) Среднее число звонков в поддержку по группам Churn:
Churn
False    1.449825
True     2.229814

6) Crosstab(Customer service calls × Churn) с долей оттока по каждому числу звонков.
   Общая доля оттока = 0.145. Порог(и) звонков, где churn_rate > 0.40: [4, 5, 6, 7, 8, 9].

7) Crosstab(International plan × Churn) с долей оттока:
Churn               False  True  churn_rate
International plan                         
No                   2664   346    0.114950
Yes                   186   137    0.424149

8) Прогноз: Predicted churn = (Customer service calls ≥ 4) OR (International plan == 'Yes').
   Матрица ошибок: FP=306, FN=227, TN=2544, TP=256.
   Ошибка I рода (ложноположительные): 0.107
   Ошибка II рода (ложноотрицательные): 0.470

Все таблицы и топ‑10 по длительности звонков печатаются при запуске.
"""

import pandas as pd


use_cols = [
    "State", "Area code", "International plan", "Number vmail messages",
    "Total day minutes", "Total day calls",
    "Total eve minutes", "Total eve calls",
    "Total night minutes", "Total night calls",
    "Customer service calls", "Churn"
]
df = pd.read_csv("telecom_churn.csv", usecols=use_cols)

print("=== Пропуски по столбцам ===")
print(df.isna().sum())

print("\n=== Описательная статистика (числовые) ===")
print(df.describe())

print("\n=== value_counts по Churn ===")
counts = df["Churn"].value_counts(dropna=False)
print(counts)
print("\nПроценты (%):")
print((counts / len(df) * 100).round(2))

total_minutes = df["Total day minutes"] + df["Total eve minutes"] + df["Total night minutes"]
total_calls = df["Total day calls"] + df["Total eve calls"] + df["Total night calls"]
df["Avg call duration"] = (total_minutes / total_calls.replace({0: pd.NA})).astype(float)

print("\n=== Топ-10 по средней длительности звонка ===")
print(df.sort_values("Avg call duration", ascending=False).head(10))

print("\n=== Средняя длительность звонка по группам Churn ===")
print(df.groupby("Churn")["Avg call duration"].mean())

print("\n=== Среднее число звонков в поддержку по группам Churn ===")
print(df.groupby("Churn")["Customer service calls"].mean())

ct_calls = pd.crosstab(df["Customer service calls"], df["Churn"])
ct_calls["churn_rate"] = ct_calls.get(True, 0) / ct_calls.sum(axis=1)
print("\n=== Crosstab: Customer service calls × Churn ===")
print(ct_calls)

ct_plan = pd.crosstab(df["International plan"], df["Churn"])
ct_plan["churn_rate"] = ct_plan.get(True, 0) / ct_plan.sum(axis=1)
print("\n=== Crosstab: International plan × Churn ===")
print(ct_plan)

pred = (df["Customer service calls"] >= 4) | (df["International plan"].str.lower().eq("yes"))
df["Predicted churn"] = pred

y_true = df["Churn"].astype(bool)
y_pred = df["Predicted churn"].astype(bool)
fp = int(((y_pred == True) & (y_true == False)).sum())
fn = int(((y_pred == False) & (y_true == True)).sum())
tn = int(((y_pred == False) & (y_true == False)).sum())
tp = int(((y_pred == True) & (y_true == True)).sum())
fp_rate = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
fn_rate = fn / (fn + tp) if (fn + tp) > 0 else float("nan")

print("\n=== Оценка ошибок правила (Predicted churn) ===")
print(f"FP={fp}, FN={fn}, TN={tn}, TP={tp}")
print(f"Ошибка I рода (ложноположительные): {fp_rate:.3f}")
print(f"Ошибка II рода (ложноотрицательные): {fn_rate:.3f}")
