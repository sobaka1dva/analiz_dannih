"""
ЗАДАНИЕ 2. Олимпийские игры: анализ набора athlete_events.csv

Данные: athlete_events.csv (ID, Name, Sex, Age, Height, Weight, Team, NOC, Games, Year, Season, City, Sport, Event, Medal).

2) Полнота данных (не‑NaN) и пропуски:
   - Наибольшее число пропусков в столбце: 'Medal' = 231333.
   (Подробные counts/info выводятся при выполнении скрипта.)

3) Статистика по Age/Height/Weight — describe():
                 Age         Height         Weight
count  261642.000000  210945.000000  208241.000000
mean       25.556898     175.338970      70.702393
std         6.393561      10.518462      14.348020
min        10.000000     127.000000      25.000000
25%        21.000000     168.000000      60.000000
50%        24.000000     175.000000      70.000000
75%        28.000000     183.000000      79.000000
max        97.000000     226.000000     214.000000

4) Ответы на вопросы:
   1) Самый юный участник в 1992 году: возраст = 11.
      Имя и дисциплины (возможны несколько записей):
                           Name                     Event
Carlos Bienvenido Front Barrera Rowing Men's Coxed Eights
   2) Список всех видов спорта (уникальных): всего 66 вида/видов.
      Пример первых 15: ['Aeronautics', 'Alpine Skiing', 'Alpinism', 'Archery', 'Art Competitions', 'Athletics', 'Badminton', 'Baseball', 'Basketball', 'Basque Pelota', 'Beach Volleyball', 'Biathlon', 'Bobsleigh', 'Boxing', 'Canoeing']
   3) Средний рост теннисисток (F, Tennis) на ОИ‑2000: 171.79 см.
   4) Количество золотых медалей Китая (CHN) в настольном теннисе на ОИ‑2008: 8.
   5) Число видов спорта на летних ОИ: 1988 — 27, 2004 — 34. Изменение: +7.
   6) Построение гистограммы возраста мужчин‑керлингистов (2014) — выполняется при запуске (matplotlib).
   7) Зимняя ОИ‑2006: по странам (NOC) — количество медалей и средний возраст (только страны с ≥1 медалью).
   8) Зимняя ОИ‑2006: сводная таблица (pivot) по числу медалей каждого достоинства на страну, без NaN (fill_value=0).

Скрипт печатает ключевые результаты, строит гистограмму для п. 6 и выдаёт таблицы для п. 7–8.
"""

import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("athlete_events.csv")

# 2) Полнота/пропуски
print("=== Количество непустых значений по столбцам ===")
print(df.count())
print("\n=== Число пропусков по столбцам ===")
print(df.isna().sum())

# 3) describe по Age/Height/Weight
print("\n=== describe(Age, Height, Weight) ===")
print(df[["Age", "Height", "Weight"]].describe())

# 4.1) Самый юный участник в 1992
y1992 = df[df["Year"] == 1992]
min_age = int(y1992["Age"].min())
print(f"\nСамый юный участник в 1992: {min_age} лет")
print(y1992[y1992["Age"] == min_age][["Name", "Event"]].drop_duplicates())

# 4.2) Все виды спорта
sports = sorted(df["Sport"].dropna().unique().tolist())
print(f"\nВсего уникальных видов спорта: {len(sports)}")
print("Первые 15:", sports[:15])

# 4.3) Средний рост теннисисток 2000 г.
mask = (df["Year"] == 2000) & (df["Sex"] == "F") & (df["Sport"] == "Tennis")
print("\nСредний рост теннисисток (2000):", float(df.loc[mask, "Height"].mean()))

# 4.4) Золото Китая в настольном теннисе, 2008
mask_tt = (df["Year"] == 2008) & (df["Sport"] == "Table Tennis") & (df["Medal"] == "Gold") & (df["NOC"] == "CHN")
print("\nЗолотых медалей Китая (настольный теннис, 2008):", int(mask_tt.sum()))

# 4.5) Изменение количества видов спорта 1988 vs 2004 (Summer)
s88 = df[(df["Season"] == "Summer") & (df["Year"] == 1988)]["Sport"].nunique()
s04 = df[(df["Season"] == "Summer") & (df["Year"] == 2004)]["Sport"].nunique()
print(f"\nЛетние ОИ: 1988 — {s88}, 2004 — {s04}. Изменение: {s04 - s88:+d}")

# 4.6) Гистограмма возраста мужчин-керлингистов (2014)
mask_curl_2014_m = (df["Year"] == 2014) & (df["Sport"] == "Curling") & (df["Sex"] == "M")
ages = df.loc[mask_curl_2014_m, "Age"].dropna()
plt.figure()
plt.hist(ages)
plt.xlabel("Возраст (годы)")
plt.ylabel("Частота")
plt.title("Распределение возраста мужчин-керлингистов (ОИ 2014)")
plt.show()

# 4.7) Зимняя 2006: по странам — число медалей и средний возраст (только с ≥1 медалью)
w06 = df[(df["Season"] == "Winter") & (df["Year"] == 2006)]
w06_med = w06.dropna(subset=["Medal"])
grouped = (
    w06_med.groupby("NOC")
    .agg(medals=("Medal", "count"), avg_age=("Age", "mean"))
    .sort_values("medals", ascending=False)
)
print("\n=== Зимняя 2006: медали и средний возраст по странам (только с ≥1 медалью) ===")
print(grouped)

# 4.8) Зимняя 2006: pivot по медалям (без NaN)
pivot = w06_med.pivot_table(index="NOC", columns="Medal", values="ID", aggfunc="count", fill_value=0).reset_index()
print("\n=== Зимняя 2006: сводная таблица медалей (pivot) ===")
print(pivot)
