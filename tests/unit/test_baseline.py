import pytest
from credit_risk.data.loader import load_german_credit
from credit_risk.models.baseline import build_baseline_pipeline, evaluate_model, build_lightgbm_pipeline
from credit_risk.features.prepare import split_features_target
from sklearn.model_selection import train_test_split

def test_build_baseline_pipeline():
    df = load_german_credit()
    x, y = split_features_target(df)
    num_cols = x.select_dtypes(include='number').columns.tolist()
    cat_cols = x.select_dtypes(include='object').columns.tolist()

    pipeline = build_baseline_pipeline(num_cols, cat_cols)

    pipeline.fit(x, y)


    predict = pipeline.predict(x).shape
    assert predict[0] == x.shape[0] #для теста не кртитчно обучение и предикт на x. Цель лишь удостоверится что всё сработало


def test_evaluate_model():
    df = load_german_credit()

    x, y = split_features_target(df)  # ИЗМЕНЕНО: перенесено ВЫШЕ train_test_split — сначала нужно получить x, y, потом их разбивать

    X_train, X_test, Y_train, Y_test = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=42
    )  # ИЗМЕНЕНО: раньше вызывалось без аргументов — теперь передаём x, y и параметры разбиения

    num_cols = X_train.select_dtypes(include='number').columns.tolist()  # ИЗМЕНЕНО: было x.select_dtypes(...) — теперь именно X_train, а не весь x
    cat_cols = X_train.select_dtypes(include='object').columns.tolist()  # аналогично

    pipeline = build_baseline_pipeline(num_cols, cat_cols)

    pipeline.fit(X_train, Y_train)  # ИЗМЕНЕНО: убрано лишнее присваивание "model =" — fit меняет pipeline на месте

    auc = evaluate_model(pipeline, X_test, Y_test)  # ИЗМЕНЕНО: сохраняем результат в переменную, а не просто вызываем и теряем значение

    assert 0 <= auc <= 1  # ДОБАВЛЕНО: сам assert — раньше его не было вообще, тест ничего не проверял


def test_lightgbm_vs_baseline():
    df = load_german_credit()

    x, y = split_features_target(df)  

    X_train, X_test, Y_train, Y_test = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=42
    )  

    num_cols = X_train.select_dtypes(include='number').columns.tolist() 
    cat_cols = X_train.select_dtypes(include='object').columns.tolist()

    pipeline = build_baseline_pipeline(num_cols, cat_cols)
    pipeline.fit(X_train, Y_train)
    auc_logreg = evaluate_model(pipeline, X_test, Y_test)

    pipeline = build_lightgbm_pipeline(num_cols, cat_cols)
    pipeline.fit(X_train, Y_train)
    auc_lightgmb = evaluate_model(pipeline, X_test, Y_test)

    print(auc_logreg)
    print(auc_lightgmb)

    assert 0 <= auc_lightgmb <= 1