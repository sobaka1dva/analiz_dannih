import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

# -------------------------------------------
# 1. Настройки
# -------------------------------------------
CSV_PATH = "weather1.csv"           # путь к файлу
OUT_DIR = "lab05_results"           # папка для сохранения графиков
os.makedirs(OUT_DIR, exist_ok=True)

# -------------------------------------------
# 2. Загрузка данных (cp1251, ; , " )
# -------------------------------------------
df = pd.read_csv(CSV_PATH, sep=';', encoding='cp1251', quotechar='"', low_memory=False)

# Переименуем первый столбец
df = df.rename(columns={df.columns[0]: "local_time"})

# Точные столбцы, которые нужны в ЛР
needed_cols = ["local_time", "T", "P", "U", "Ff", "N", "H", "VV"]

# Если какого-то столбца нет — добавим пустой (чтобы код не падал)
for col in needed_cols:
    if col not in df.columns:
        df[col] = np.nan

# -------------------------------------------
# 3. Преобразование типов
# -------------------------------------------
# local_time → datetime
df["local_time_parsed"] = pd.to_datetime(df["local_time"], errors='coerce', dayfirst=True)

# Приведение остальных столбцов к числам
for col in ["T", "P", "U", "Ff", "N", "H", "VV"]:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# -------------------------------------------
# 4. Scatter: T vs U
# -------------------------------------------
plt.figure(figsize=(7, 5))
plt.scatter(df["T"], df["U"], alpha=0.6)
plt.xlabel("Температура T (°C)")
plt.ylabel("Влажность U (%)")
plt.title("Диаграмма рассеяния: T vs U")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/scatter_T_U.png", dpi=150)
plt.close()

# -------------------------------------------
# 5. Scatter с раскраской по облачности N (100 — синий, остальные — красный)
# -------------------------------------------
mask100 = df["N"] == 100

plt.figure(figsize=(7, 5))
plt.scatter(df.loc[mask100, "T"], df.loc[mask100, "U"], alpha=0.7, c="blue", label="N = 100")
plt.scatter(df.loc[~mask100, "T"], df.loc[~mask100, "U"], alpha=0.5, c="red", label="N ≠ 100")
plt.xlabel("Температура T (°C)")
plt.ylabel("Влажность U (%)")
plt.title("T vs U с выделением облачности N = 100")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/scatter_T_U_colored.png", dpi=150)
plt.close()

# -------------------------------------------
# 6. Линейный график температуры по времени
# -------------------------------------------
df_time = df.dropna(subset=["local_time_parsed", "T"]).sort_values("local_time_parsed")

plt.figure(figsize=(10, 5))
plt.plot(df_time["local_time_parsed"], df_time["T"], marker="o", markersize=3)
plt.xlabel("Время")
plt.ylabel("Температура T (°C)")
plt.title("Температура по времени")
plt.gcf().autofmt_xdate()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/line_temperature.png", dpi=150)
plt.close()

# -------------------------------------------
# 7. Среднемесячная температура (bar chart)
# -------------------------------------------
df["month"] = df["local_time_parsed"].dt.month
monthly_avg = df.dropna(subset=["month", "T"]).groupby("month")["T"].mean()

plt.figure(figsize=(8, 5))
sns.barplot(x=monthly_avg.index, y=monthly_avg.values)
plt.xlabel("Месяц")
plt.ylabel("Средняя температура (°C)")
plt.title("Среднемесячная температура")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/bar_monthly_avg_T.png", dpi=150)
plt.close()

# -------------------------------------------
# 8. Горизонтальная диаграмма количества наблюдений по облачности N
# -------------------------------------------
counts_N = df["N"].value_counts().sort_index()

plt.figure(figsize=(8, 6))
plt.barh(counts_N.index.astype(str), counts_N.values)
plt.xlabel("Количество наблюдений")
plt.ylabel("N — облачность")
plt.title("Количество наблюдений по облачности N")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/hbar_cloudiness_counts.png", dpi=150)
plt.close()

# -------------------------------------------
# 9. Гистограмма температуры T (10 корзин)
# -------------------------------------------
plt.figure(figsize=(7, 5))
plt.hist(df["T"].dropna(), bins=10, edgecolor="black")
plt.xlabel("Температура T (°C)")
plt.ylabel("Частота")
plt.title("Гистограмма температуры (10 интервалов)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/hist_T_10bins.png", dpi=150)
plt.close()

# -------------------------------------------
# 10. Boxplot давления P в зависимости от видимости VV
# -------------------------------------------

group1 = df[df["VV"] < 5]["P"].dropna()
group2 = df[(df["VV"] >= 5) & (df["VV"] <= 15)]["P"].dropna()
group3 = df[df["VV"] > 15]["P"].dropna()

plt.figure(figsize=(8, 6))
plt.boxplot([group1, group2, group3],
            labels=["VV < 5", "5 ≤ VV ≤ 15", "VV > 15"])
plt.ylabel("Давление P (мм рт. ст.)")
plt.title("Распределение давления по группам видимости")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/boxplots_P_by_visibility.png", dpi=150)
plt.close()

# -------------------------------------------
# 11. Pie chart по основанию облаков (H)
# -------------------------------------------
h_counts = df["H"].fillna("NaN").astype(str).value_counts()

# если слишком много категорий ‒ оставим топ-6
if len(h_counts) > 6:
    h_plot = h_counts.iloc[:6].copy()
    h_plot["other"] = h_counts.iloc[6:].sum()
else:
    h_plot = h_counts

plt.figure(figsize=(7, 7))
plt.pie(h_plot.values, labels=h_plot.index,
        autopct="%1.1f%%", startangle=90)
plt.title("Распределение высоты основания облаков H")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/pie_H.png", dpi=150)
plt.close()

print(OUT_DIR)
