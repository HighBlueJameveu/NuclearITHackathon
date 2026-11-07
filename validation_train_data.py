"""
Валидация обучающей выборки для задачи предсказания SI VACV.

Проверки:
1. Наличие обязательных колонок.
2. Валидность всех SMILES через RDKit.
3. Согласованность SI = CC50 / IC50.
4. Положительность IC50, CC50, SI.
5. Отсутствие пропусков и бесконечностей.
6. Отсутствие дубликатов SMILES.
7. Разумные диапазоны значений (грубая проверка на выбросы).
8. Наличие DOI (предупреждение, а не ошибка).

Запуск:
    python validate_training_set.py --input train_vacv_clean.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

# Отключаем спам RDKit в консоль
RDLogger.DisableLog("rdApp.*")


# ---------- Настройки ----------
REQUIRED_COLUMNS = ["SMILES", "IC50_uM", "CC50_uM", "SI_VACV"]

# Допустимое относительное расхождение SI и CC50/IC50
SI_REL_TOLERANCE = 0.01       # 1%

# Пороги «разумности» (мягкие, только для предупреждений)
IC50_MAX_REASONABLE = 1e5     # µM
CC50_MAX_REASONABLE = 1e5
SI_MAX_REASONABLE = 1e6

# Минимальное количество строк для обучения
MIN_ROWS = 10


# ---------- Утилиты ----------
def log(msg, level="INFO"):
    prefix = {
        "INFO": "[INFO]",
        "OK": "[ OK ]",
        "WARN": "[WARN]",
        "FAIL": "[FAIL]",
    }.get(level, "[????]")
    print(f"{prefix} {msg}")


def check_columns(df, errors, warnings):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        errors.append(f"Отсутствуют обязательные колонки: {missing}")
    else:
        log(f"Все обязательные колонки на месте: {REQUIRED_COLUMNS}", "OK")

    # DOI — не обязателен, но полезен
    if "doi" not in df.columns:
        warnings.append("Колонка 'doi' отсутствует — на защите будет сложно показать источник.")


def check_smiles(df, errors, warnings):
    if "SMILES" not in df.columns:
        return

    smiles_series = df["SMILES"].astype(str)
    invalid = []
    for idx, smi in smiles_series.items():
        if Chem.MolFromSmiles(smi) is None:
            invalid.append((idx, smi))

    if invalid:
        errors.append(
            f"Невалидных SMILES: {len(invalid)}. "
            f"Примеры: {[s for _, s in invalid[:3]]}"
        )
    else:
        log(f"Все SMILES валидны ({len(df)} шт.)", "OK")


def check_missing_and_infinite(df, errors):
    for col in ["IC50_uM", "CC50_uM", "SI_VACV"]:
        if col not in df.columns:
            continue

        n_nan = df[col].isna().sum()
        if n_nan > 0:
            errors.append(f"В колонке {col} пропусков: {n_nan}")

        # Бесконечности (после конвертации в float)
        numeric = pd.to_numeric(df[col], errors="coerce")
        n_inf = np.isinf(numeric).sum()
        if n_inf > 0:
            errors.append(f"В колонке {col} бесконечностей: {n_inf}")

    log("Проверка пропусков и бесконечностей завершена", "OK")


def check_positivity(df, errors):
    for col in ["IC50_uM", "CC50_uM", "SI_VACV"]:
        if col not in df.columns:
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        non_positive = (numeric <= 0).sum()
        if non_positive > 0:
            errors.append(f"В колонке {col} неположительных значений: {non_positive}")

    log("Проверка положительности завершена", "OK")


def check_si_consistency(df, errors):
    if not {"IC50_uM", "CC50_uM", "SI_VACV"}.issubset(df.columns):
        return

    ic50 = pd.to_numeric(df["IC50_uM"], errors="coerce")
    cc50 = pd.to_numeric(df["CC50_uM"], errors="coerce")
    si = pd.to_numeric(df["SI_VACV"], errors="coerce")

    si_check = cc50 / ic50
    rel_err = (si_check - si).abs() / si

    bad = rel_err > SI_REL_TOLERANCE
    n_bad = bad.sum()

    if n_bad > 0:
        examples = df.loc[bad, ["IC50_uM", "CC50_uM", "SI_VACV"]].head(3)
        errors.append(
            f"Строк с расхождением SI = CC50/IC50 более чем на "
            f"{SI_REL_TOLERANCE * 100:.1f}%: {n_bad}. Примеры:\n{examples}"
        )
    else:
        max_err = rel_err.max()
        log(
            f"SI согласован с CC50/IC50 (макс. расхождение {max_err * 100:.4f}%)",
            "OK",
        )


def check_duplicates(df, errors):
    if "SMILES" not in df.columns:
        return

    n_dups = df["SMILES"].duplicated().sum()
    if n_dups > 0:
        errors.append(f"Дубликатов SMILES: {n_dups}")
    else:
        log("Дубликатов SMILES нет", "OK")


def check_ranges(df, warnings):
    if "IC50_uM" in df.columns:
        mx = pd.to_numeric(df["IC50_uM"], errors="coerce").max()
        if mx > IC50_MAX_REASONABLE:
            warnings.append(
                f"IC50_uM: подозрительно большое значение {mx} "
                f"(порог {IC50_MAX_REASONABLE})."
            )

    if "CC50_uM" in df.columns:
        mx = pd.to_numeric(df["CC50_uM"], errors="coerce").max()
        if mx > CC50_MAX_REASONABLE:
            warnings.append(
                f"CC50_uM: подозрительно большое значение {mx} "
                f"(порог {CC50_MAX_REASONABLE})."
            )

    if "SI_VACV" in df.columns:
        mx = pd.to_numeric(df["SI_VACV"], errors="coerce").max()
        if mx > SI_MAX_REASONABLE:
            warnings.append(
                f"SI_VACV: подозрительно большое значение {mx} "
                f"(порог {SI_MAX_REASONABLE})."
            )

    log("Проверка диапазонов завершена", "OK")


def check_doi(df, warnings):
    if "doi" not in df.columns:
        return
    n_missing = df["doi"].isna().sum()
    n_total = len(df)
    if n_missing > 0:
        warnings.append(
            f"Строк без DOI: {n_missing} из {n_total}. "
            f"Для защиты лучше указать источник для каждой строки."
        )
    else:
        log("DOI заполнен для всех строк", "OK")


def check_min_rows(df, errors):
    if len(df) < MIN_ROWS:
        errors.append(
            f"Слишком мало строк: {len(df)} < {MIN_ROWS}. "
            f"Модель не сможет обучиться."
        )
    else:
        log(f"Достаточно строк для обучения: {len(df)}", "OK")


# ---------- Отчёт ----------
def save_report(df, errors, warnings, path):
    lines = []
    lines.append("=" * 70)
    lines.append("ОТЧЁТ ВАЛИДАЦИИ ОБУЧАЮЩЕЙ ВЫБОРКИ")
    lines.append("=" * 70)
    lines.append(f"Файл: {path}")
    lines.append(f"Строк: {len(df)}")
    lines.append(f"Колонок: {len(df.columns)}")
    lines.append("")

    lines.append("КОЛОНКИ:")
    for col in df.columns:
        lines.append(f"  - {col}: dtype={df[col].dtype}")
    lines.append("")

    if "IC50_uM" in df.columns:
        lines.append("ОПИСАТЕЛЬНЫЕ СТАТИСТИКИ:")
        desc = df[["IC50_uM", "CC50_uM", "SI_VACV"]].describe()
        lines.append(desc.to_string())
        lines.append("")

    lines.append(f"ОШИБОК: {len(errors)}")
    for e in errors:
        lines.append(f"  [FAIL] {e}")
    lines.append("")

    lines.append(f"ПРЕДУПРЕЖДЕНИЙ: {len(warnings)}")
    for w in warnings:
        lines.append(f"  [WARN] {w}")
    lines.append("")

    lines.append("=" * 70)
    if errors:
        lines.append("РЕЗУЛЬТАТ: НЕ ПРОЙДЕНО. Данные нельзя использовать для обучения.")
    else:
        lines.append("РЕЗУЛЬТАТ: ПРОЙДЕНО. Данные можно использовать для обучения.")
    lines.append("=" * 70)

    report = "\n".join(lines)
    with open("validation_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    return report


# ---------- Точка входа ----------
def main():
    parser = argparse.ArgumentParser(description="Валидация обучающей выборки")
    parser.add_argument(
        "--input",
        default="train_vacv_clean.csv",
        help="Путь к CSV с обучающей выборкой",
    )
    args = parser.parse_args()

    if not Path(args.input).exists():
        log(f"Файл не найден: {args.input}", "FAIL")
        sys.exit(1)

    print("=" * 70)
    print(f"Валидация файла: {args.input}")
    print("=" * 70)

    df = pd.read_csv(args.input)

    errors = []
    warnings = []

    # 1. Колонки
    check_columns(df, errors, warnings)

    # 2. SMILES
    check_smiles(df, errors, warnings)

    # 3. Пропуски и бесконечности
    check_missing_and_infinite(df, errors)

    # 4. Положительность
    check_positivity(df, errors)

    # 5. SI = CC50 / IC50
    check_si_consistency(df, errors)

    # 6. Дубликаты
    check_duplicates(df, errors)

    # 7. Диапазоны
    check_ranges(df, warnings)

    # 8. DOI
    check_doi(df, warnings)

    # 9. Минимум строк
    check_min_rows(df, errors)

    # Отчёт
    print("\n" + "=" * 70)
    print("ИТОГ")
    print("=" * 70)

    if warnings:
        print("\nПредупреждения:")
        for w in warnings:
            log(w, "WARN")

    if errors:
        print("\nОшибки:")
        for e in errors:
            log(e, "FAIL")
        print()
        log("Данные НЕ ПРОШЛИ валидацию. Исправьте ошибки перед обучением.", "FAIL")
        save_report(df, errors, warnings, args.input)
        sys.exit(1)

    log("Все проверки пройдены. Данные можно использовать для обучения.", "OK")
    save_report(df, errors, warnings, args.input)
    print("\nПолный отчёт сохранён в validation_report.txt")


if __name__ == "__main__":
    main()