# -*- coding: utf-8 -*-
"""
Кластеризация звёзд и поиск центроидов (по условию = медоиды).
Вход: CSV с десятичной запятой/точкой и разделителем ';' или ','.

Настрой:
- INPUT_CSV: путь к файлу
"""

from pathlib import Path
from typing import List, Tuple, Optional
import csv

import numpy as np
from sklearn.cluster import KMeans


# ====== НАСТРОЙКИ ======
INPUT_CSV = Path("27_B_17834.csv")   # <-- укажи свой путь
K = 3
SEED = 42
SORT_OUTPUT_BY_MEDOID = True
# =======================


def _clean_cell(s: str) -> str:
    return s.replace("\ufeff", "").strip()


def _to_float(s: str) -> float:
    s = _clean_cell(s)
    s = s.replace(",", ".")  # десятичная запятая -> точка
    return float(s)


def _try_parse_rows(sample_lines: List[str], delimiter: str) -> Tuple[int, bool, Optional[int], Optional[int]]:
    """
    Возвращает:
    - score: сколько строк успешно распарсили как (x, y)
    - has_header: найден заголовок x/y
    - ix, iy: индексы колонок x/y если заголовок найден
    """
    score = 0
    has_header = False
    ix = iy = None

    reader = csv.reader(sample_lines, delimiter=delimiter)
    rows = [row for row in reader if row and any(_clean_cell(c) for c in row)]
    if not rows:
        return 0, False, None, None

    first = [_clean_cell(c) for c in rows[0]]
    lower = [c.lower() for c in first]

    if ("x" in lower) and ("y" in lower):
        has_header = True
        ix = lower.index("x")
        iy = lower.index("y")
        data_rows = rows[1:]
        for r in data_rows:
            if len(r) <= max(ix, iy):
                continue
            try:
                _to_float(r[ix])
                _to_float(r[iy])
                score += 1
            except Exception:
                pass
        return score, has_header, ix, iy

    # заголовка нет: пробуем первые 2 столбца как числа
    for r in rows:
        if len(r) < 2:
            continue
        try:
            _to_float(r[0])
            _to_float(r[1])
            score += 1
        except Exception:
            pass

    return score, False, None, None


def detect_csv_format(path: Path) -> Tuple[str, bool, Optional[int], Optional[int]]:
    """
    Выбирает разделитель по реальной распарсиваемости чисел.
    При равенстве баллов приоритет: ';' затем ',' затем '\\t'
    """
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    sample = [ln for ln in lines if ln.strip()][:30]
    if not sample:
        raise ValueError("CSV пустой.")

    candidates = [";", ",", "\t"]
    best = None

    for d in candidates:
        score, has_header, ix, iy = _try_parse_rows(sample, d)
        item = (score, d, has_header, ix, iy)
        if best is None:
            best = item
            continue
        # выбираем по score, затем по приоритету разделителя
        if score > best[0]:
            best = item
        elif score == best[0]:
            # приоритет кандидатов по порядку в candidates
            if candidates.index(d) < candidates.index(best[1]):
                best = item

    if best is None or best[0] == 0:
        raise ValueError("Не удалось определить формат CSV: числа не читаются как x/y.")

    _, delimiter, has_header, ix, iy = best
    return delimiter, has_header, ix, iy


def load_points_csv(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path.resolve()}")

    delimiter, has_header_hint, ix_hint, iy_hint = detect_csv_format(path)

    pts: List[Tuple[float, float]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f, delimiter=delimiter)

        try:
            first_row_raw = next(reader)
        except StopIteration:
            raise ValueError("CSV пустой.")

        first_row = [_clean_cell(c) for c in first_row_raw]

        # финальная проверка заголовка
        lower = [c.lower() for c in first_row]
        has_header = ("x" in lower) and ("y" in lower)

        if has_header:
            ix = lower.index("x")
            iy = lower.index("y")
            start_line_no = 2
        elif has_header_hint and ix_hint is not None and iy_hint is not None:
            # если в сэмпле нашли заголовок, но здесь не нашли из-за пробелов/вариантов,
            # используем подсказку
            ix, iy = ix_hint, iy_hint
            has_header = True
            start_line_no = 2
        else:
            # первая строка — данные
            if len(first_row) < 2:
                raise ValueError(f"Первая строка слишком короткая: {first_row_raw}")
            try:
                x0 = _to_float(first_row[0])
                y0 = _to_float(first_row[1])
            except Exception as e:
                raise ValueError(f"Первая строка не распознана как данные: {first_row}") from e
            pts.append((x0, y0))
            start_line_no = 2

        # читаем остальное
        for line_no, row in enumerate(reader, start=start_line_no):
            if not row or all(_clean_cell(c) == "" for c in row):
                continue

            if has_header:
                if len(row) <= max(ix, iy):
                    raise ValueError(f"Строка {line_no}: не хватает столбцов: {row}")
                try:
                    x = _to_float(row[ix])
                    y = _to_float(row[iy])
                except Exception as e:
                    raise ValueError(f"Строка {line_no}: не удалось прочитать x/y: {row}") from e
            else:
                if len(row) < 2:
                    raise ValueError(f"Строка {line_no}: нужно минимум 2 столбца: {row}")
                try:
                    x = _to_float(row[0])
                    y = _to_float(row[1])
                except Exception as e:
                    raise ValueError(f"Строка {line_no}: не удалось прочитать числа: {row}") from e

            pts.append((x, y))

    if not pts:
        raise ValueError("В CSV нет точек.")

    return np.array(pts, dtype=float)


def medoid(points: np.ndarray) -> np.ndarray:
    diffs = points[:, None, :] - points[None, :, :]  # (m, m, 2)
    dists = np.sqrt((diffs ** 2).sum(axis=2))        # (m, m)
    sums = dists.sum(axis=1)                         # (m,)
    return points[int(np.argmin(sums))]


def main() -> None:
    pts = load_points_csv(INPUT_CSV)

    kmeans = KMeans(n_clusters=K, n_init=50, random_state=SEED)
    labels = kmeans.fit_predict(pts)

    results = []
    for cid in range(K):
        cpts = pts[labels == cid]
        m = medoid(cpts)
        results.append((cid + 1, int(cpts.shape[0]), float(m[0]), float(m[1])))

    if SORT_OUTPUT_BY_MEDOID:
        results.sort(key=lambda t: (t[2], t[3]))

    print("Центроиды (по условию: медоиды). Формат: кластер | размер | x | y")
    for cid, size, x, y in results:
        print(f"{cid:>7} | {size:>5} | {x:>10.6f} | {y:>10.6f}")


if __name__ == "__main__":
    main()
