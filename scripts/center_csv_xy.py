"""Center motion CSV root XY columns around the first data frame.

This is useful before converting motion CSV files to NPZ for trampoline
training, where the trampoline is spawned at the environment origin.

Example:

    python scripts/center_csv_xy.py datag1/wbt_g1_motion.csv --backup
"""

from __future__ import annotations

import argparse
import csv
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CenterResult:
    path: Path
    changed: bool
    rows: int
    first_x: float
    first_y: float
    min_x: float
    max_x: float
    min_y: float
    max_y: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Center one motion CSV by subtracting the first row's root x/y.")
    parser.add_argument("csv_path", type=Path, help="Motion CSV file to center in place.")
    parser.add_argument("--x-col", type=int, default=0, help="Zero-based CSV column index for root x.")
    parser.add_argument("--y-col", type=int, default=1, help="Zero-based CSV column index for root y.")
    parser.add_argument("--precision", type=int, default=8, help="Decimal places used for rewritten x/y columns.")
    parser.add_argument("--backup", action="store_true", help="Write a .bak copy before replacing the CSV.")
    return parser.parse_args()


def parse_float(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def read_rows(path: Path) -> list[list[str]]:
    with path.open("r", newline="") as stream:
        return list(csv.reader(stream))


def first_numeric_row(rows: list[list[str]], x_col: int, y_col: int) -> tuple[int, float, float]:
    required_cols = max(x_col, y_col) + 1
    for row_index, row in enumerate(rows):
        if len(row) < required_cols:
            continue
        x = parse_float(row[x_col])
        y = parse_float(row[y_col])
        if x is not None and y is not None:
            return row_index, x, y
    raise ValueError("No data row with numeric x/y columns was found.")


def center_rows(
    rows: list[list[str]],
    *,
    path: Path,
    x_col: int,
    y_col: int,
    precision: int,
) -> tuple[list[list[str]], CenterResult]:
    first_row_index, first_x, first_y = first_numeric_row(rows, x_col, y_col)
    required_cols = max(x_col, y_col) + 1
    centered_rows: list[list[str]] = []
    min_x = float("inf")
    max_x = float("-inf")
    min_y = float("inf")
    max_y = float("-inf")
    data_rows = 0

    for row_index, row in enumerate(rows):
        if row_index < first_row_index or not row:
            centered_rows.append(row)
            continue
        if len(row) < required_cols:
            raise ValueError(f"Row {row_index + 1} has {len(row)} columns, expected at least {required_cols}.")

        x = parse_float(row[x_col])
        y = parse_float(row[y_col])
        if x is None or y is None:
            raise ValueError(f"Row {row_index + 1} has non-numeric x/y values: {row[x_col]!r}, {row[y_col]!r}")

        centered_x = x - first_x
        centered_y = y - first_y
        min_x = min(min_x, centered_x)
        max_x = max(max_x, centered_x)
        min_y = min(min_y, centered_y)
        max_y = max(max_y, centered_y)
        data_rows += 1

        centered_row = list(row)
        centered_row[x_col] = f"{centered_x:.{precision}f}"
        centered_row[y_col] = f"{centered_y:.{precision}f}"
        centered_rows.append(centered_row)

    result = CenterResult(
        path=path,
        changed=centered_rows != rows,
        rows=data_rows,
        first_x=first_x,
        first_y=first_y,
        min_x=min_x,
        max_x=max_x,
        min_y=min_y,
        max_y=max_y,
    )
    return centered_rows, result


def write_rows_atomic(path: Path, rows: list[list[str]], backup: bool) -> None:
    if backup:
        shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))

    with tempfile.NamedTemporaryFile("w", newline="", dir=path.parent, delete=False) as stream:
        temp_path = Path(stream.name)
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerows(rows)
    temp_path.replace(path)


def print_result(result: CenterResult) -> None:
    action = "updated" if result.changed else "already centered"
    print(
        f"{result.path}: {action}; rows={result.rows}; "
        f"subtract=({result.first_x:.8f}, {result.first_y:.8f}); "
        f"new_x=[{result.min_x:.8f}, {result.max_x:.8f}], "
        f"new_y=[{result.min_y:.8f}, {result.max_y:.8f}]"
    )


def main() -> None:
    args = parse_args()
    path = args.csv_path.expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"CSV path does not exist or is not a file: {path}")
    if path.suffix.lower() != ".csv":
        raise ValueError(f"Expected a .csv file, got: {path}")

    rows = read_rows(path)
    centered_rows, result = center_rows(
        rows,
        path=path,
        x_col=args.x_col,
        y_col=args.y_col,
        precision=args.precision,
    )
    if result.changed:
        write_rows_atomic(path, centered_rows, backup=args.backup)
    print_result(result)


if __name__ == "__main__":
    main()
