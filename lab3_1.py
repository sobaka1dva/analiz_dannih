import numpy as np

print("1. Исходная матрица:")
matrix = np.random.randint(-10, 11, size=(8, 8))
print(matrix)
print()

print("2. Центральная часть матрицы:")
center_matrix = matrix[2:6, 2:6]
print(center_matrix)
print()

print("3. Матрица после удаления строк с минимальным элементом:")
min_element = np.min(matrix)
print(f"Минимальный элемент: {min_element}")

rows_with_min = np.any(matrix == min_element, axis=1)
matrix_cleaned = matrix[~rows_with_min]

print("Матрица после удаления:")
print(matrix_cleaned)
print()

print("4. Матрица после вставки строки с минимальным элементом:")
min_row = np.full((1, matrix_cleaned.shape[1]), min_element)
matrix_with_min_row = np.vstack([min_row, matrix_cleaned])
print(matrix_cleaned.shape)



print(matrix_with_min_row)
print()

print("5. Статистика элементов матрицы:")
total_sum = np.sum(matrix_with_min_row)
average = np.mean(matrix_with_min_row)

print(f"Сумма всех элементов: {total_sum}")
print(f"Среднее арифметическое: {average:.2f}")


max_el = np.max(matrix)
max_row = np.full((1, matrix.shape[1]), max_el)
print(max_row)
matrix_with_max_row = np.vstack([matrix[:4],max_row])
new_matrix_with_max_row = np.vstack([matrix_with_max_row, matrix[4:]])
print(new_matrix_with_max_row)