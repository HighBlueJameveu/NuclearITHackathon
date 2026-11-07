# Предсказание SI против вируса оспы (VACV)

Регрессия индекса селективности SI = CC50 / IC50 по SMILES. Одна модель: LightGBM.

## Структура
```
data/raw/        train.csv, test.csv, sample_submit.csv
data/external/   сюда пишет скрипт сбора данных из открытых баз (ChEMBL, PubChem, ...)
pipeline.py      полный пайплайн одной командой
src/features.py  признаки (Morgan counts + дескрипторы RDKit + CC50/IC50)
src/model.py     модель
src/train.py     scaffold-CV + обучение, сохраняет models/si_model.joblib
src/predict.py   создаёт results/submission.csv
```

## Запуск
```bash
pip install -r requirements.txt
python pipeline.py          # всё сразу: обучение + results/submission.csv
# или по шагам:
python -m src.train          # обучение и проверка
python -m src.predict        # results/submission.csv -> загрузить на Kaggle
```
Режим только по SMILES (без CC50/IC50 как признаков): `python -m src.train --no-assay`.

## Метод
- Таргет: log10(1 + SI), обучение на всех строках (включая SI = 0).
- Валидация: GroupKFold по скаффолдам Мёрко.
- В train SI = CC50/IC50 точно, поэтому модель учится на остатке от log10(1 + CC50/IC50).
  Это использует колонки CC50/IC50 из test.csv. Для чистого режима используйте `--no-assay`.

## TODO
- Скрипт сбора данных (SMILES, IC50, CC50, SI, DOI): `src/collect_data.py` + база в `data/external/`.
