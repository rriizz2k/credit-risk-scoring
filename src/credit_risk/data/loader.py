import pandas as pd


def load_german_credit(path="data/raw/german_credit.csv"):
    df = pd.read_csv(path)

    if df.shape[0] == 0:
        raise ValueError("Датасет пустой — 0 строк")

    if 'class' not in df.columns:
        raise ValueError("В датасете нет колонки 'class' — целевая переменная отсутствует")

    return df


df = load_german_credit('data/raw/german_credit.csv')
print(df.head(10))