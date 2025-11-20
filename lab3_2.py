import numpy as np

m1 = np.random.randint(10, 30, 10)
m2 = np.random.randint(10, 30, 10)

print(f"Массив m1: {m1}")
print(f"Массив m2: {m2}")
print()

print("1. Массив m3:")

m3_np = np.setxor1d(m1, m2)
print(f"m3: {m3_np}")
print()

print("2. Замена в м1 значений, кратных 2 или 3")

print(f"Исходный m1: {m1}")

mask = (m1 % 2 == 0) | (m1 % 3 == 0)

m1_mod = m1.copy()
m1_mod[mask] = 1

print(f"Измененный m1: {m1_mod}")
print()

print("3. Объединение массивов и преобразование в матрицу 4x5")

merged = np.concatenate([m1_mod, m2])
print(f"Объединенный массив: {merged}")
print(f"Длина объединенного массива: {len(merged)}")

matrix = merged.reshape(4, 5)
print("Матрица 4x5:")
print(matrix)
print()

print("4. Удаление 1 и 4 столбцов")

print("Исходная матрица:")
print(matrix)

matrix_reduced = np.delete(matrix, [0, 3], axis=1)
print("Матрица после удаления 1-го и 4-го столбцов:")
print(matrix_reduced)
print(f"Размер матрицы: {matrix_reduced.shape}")
print()

print("5. Транспонирование матрицы:")

matrix_transposed = matrix_reduced.T
print("Транспонированная матрица:")
print(matrix_transposed)
print(f"Размер транспонированной матрицы: {matrix_transposed.shape}")


mask2 = (m1 % 3 == 0) & (m1 % 4 == 0)
new_m1 = m1.copy()
new_m1[mask2] = 100

merged2 = np.concatenate([new_m1, m2])
new_matrix = merged2.reshape(2, 10)
print(new_matrix)
