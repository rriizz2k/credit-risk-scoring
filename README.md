# Credit Risk Scoring

# Credit Risk Scoring

Учебный ML-проект: предсказание вероятности невозврата кредита (credit default) на датасете German Credit, построенный по инженерным практикам, применяемым в реальных Data Science командах.

## Результаты


| Модель                                                                | ROC-AUC   | Recall (класс "дефолт") |
| --------------------------------------------------------------------------- | --------- | ---------------------------------- |
| **Logistic Regression**(все признаки +`class_weight='balanced'`) | **0.815** | **0.817**                          |
| LightGBM (все признаки +`class_weight='balanced'`)               | 0.777     | 0.55                               |

Финальная модель — Logistic Regression, несмотря на репутацию LightGBM как более мощной модели: на этом небольшом датасете (1000 строк) линейная модель с балансировкой классов показала лучший результат по обеим ключевым метрикам. Все эксперименты залогированы в MLflow.

## Стек

* **Данные и версионирование**: pandas, DVC
* **Моделирование**: scikit-learn, LightGBM
* **Трекинг экспериментов**: MLflow
* **Объяснимость**: SHAP
* **Serving**: FastAPI, Docker
* **Мониторинг**: Evidently (data drift detection)
* **Тестирование**: pytest
* **Управление зависимостями**: uv

## Структура проекта

```
credit-risk-scoring/
├── src/credit_risk/
│   ├── data/          # загрузка и валидация данных
│   ├── features/      # feature engineering, train/test split
│   ├── models/         # пайплайны моделей, обучение, метрики
│   └── api/            # FastAPI сервис
├── notebooks/          # EDA, SHAP-анализ, drift-репорты
├── tests/unit/          # unit-тесты
├── data/raw/            # датасет (версионируется через DVC, не в git)
├── Dockerfile
└── pyproject.toml
```

## Как запустить

### Установка

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

### Обучение модели

```bash
PYTHONPATH=src python3 src/credit_risk/models/train.py
```

Обучает LogReg и LightGBM, логирует метрики (ROC-AUC, recall) в MLflow, сохраняет лучшую модель в `model.pkl`.

### Тесты

```bash
pytest tests/ -v
```

### Просмотр экспериментов (MLflow)

```bash
mlflow ui
```

→ `http://localhost:5000`

### API локально

```bash
PYTHONPATH=src uvicorn credit_risk.api.main:app --reload
```

→ `http://localhost:8000/docs`

### API через Docker

```bash
docker build -t credit-risk-api .
docker run -p 8000:8000 credit-risk-api
```

## Процесс разработки

Проект велся по спринтам через GitHub Flow (feature-ветки + Pull Request на каждый этап):

1. Скелет проекта, `uv`, структура папок
2. DVC — версионирование датасета
3. Data loader с валидацией + тесты
4. EDA — баланс классов, распределения, корреляции, selection bias
5. Baseline pipeline (LogReg) + train/test split + ROC-AUC
6. LightGBM + MLflow tracking + SHAP explainability
7. Feature engineering (log-трансформация, комбинированные признаки) + `class_weight='balanced'`
8. FastAPI + Docker
9. Мониторинг дрифта данных (Evidently)

## Известные ограничения

* API не валидирует категориальные значения запроса против списка категорий, известных модели — нераспознанные строки обнуляются `OneHotEncoder`'ом вместо отката к моде (осознанно отложено)
* Датасет — German Credit (1000 строк), не отражает масштаб реальных банковских данных
* Branch protection на `main` не включён (учебный проект, соло-разработка)

## Датасет

[German Credit Data](https://www.openml.org/d/31) — 1000 заявок на кредит, 20 признаков, целевая переменная — вернул/не вернул кредит.
