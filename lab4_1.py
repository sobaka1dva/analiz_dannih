"""
ЗАДАНИЕ 1. Нормально распределённая случайная величина и базовые операции с Series/DataFrame

1) Сгенерирован массив из 1000 значений N(M=1.0, s=1.0) (rng=42) и преобразован в pandas.Series.
2) Доля значений в интервале (M-s; M+s) = (0.0; 2.0) составила: 0.6880 (~ожидание 0.6827).
3) Доля значений в интервале (M-3s; M+3s) = (-2.0; 4.0) составила: 0.9980.
   Теоретически по «правилу трёх сигм» должно быть ≈ 0.9973. Реальный результат отличается от теории из‑за конечной выборки и случайных флуктуаций, но в целом согласуется.
4) Преобразование x → sqrt(x): numpy предупреждает ('invalid value encountered in sqrt'). Причина: отрицательные x дают невещественный корень; таким элементам присваивается NaN.
   Количество NaN после преобразования: 171.
5) Среднее значение после преобразования (игнорируя NaN): 1.069517.
6) Сформирован DataFrame с колонками: number (исходные значения) и root (sqrt). Выводятся первые 6 строк.
7) Фильтр по query: 1.8 ≤ root ≤ 1.9.

Скрипт печатает сводки, .head(6) датафрейма и результат выборки.
"""

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
M, s = 1.0, 1.0
ser = pd.Series(rng.normal(loc=M, scale=s, size=1000), name="number")

print(ser)

# Доли в (M-s, M+s) и (M-3s, M+3s)
frac_1s = ((ser > (M - s)) & (ser < (M + s))).mean()
frac_3s = ((ser > (M - 3*s)) & (ser < (M + 3*s))).mean()
print(f"Доля в (M-s, M+s): {frac_1s:.4f}")
print(f"Доля в (M-3s, M+3s): {frac_3s:.4f} (теория ≈ 0.9973)")

# sqrt (может вызвать предупреждение для отрицательных значений)
root = pd.Series(np.sqrt(ser), name="root")
print("Количество NaN после sqrt:", int(root.isna().sum()))

# Среднее по root без NaN
print("Среднее по sqrt:", float(root.mean(skipna=True)))

# DataFrame и вывод первых 6 строк
df = pd.concat([ser, root], axis=1)
print("\nПервые 6 строк:")
print(df.head(6))

# Query по диапазону корня
res = df.query("root >= 1.8 and root <= 1.9")
print("\nЗаписи c 1.8 ≤ root ≤ 1.9:")
print(res)
