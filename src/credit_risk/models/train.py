from credit_risk.data.loader import load_german_credit
from credit_risk.models.baseline import build_baseline_pipeline, evaluate_model
from credit_risk.features.prepare import split_features_target
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    df = load_german_credit()
    x, y = split_features_target(df)
    X_train, X_test, Y_train, Y_test = train_test_split(
    x, y, test_size=0.2, stratify=y, random_state=42
    )  # ИЗМЕНЕНО: раньше вызывалось без аргументов — теперь передаём x, y и параметры разбиения

    num_cols = X_train.select_dtypes(include='number').columns.tolist()  # ИЗМЕНЕНО: было x.select_dtypes(...) — теперь именно X_train, а не весь x
    cat_cols = X_train.select_dtypes(include='object').columns.tolist()  # аналогично

    pipeline = build_baseline_pipeline(num_cols, cat_cols)

    pipeline.fit(X_train, Y_train)  # ИЗМЕНЕНО: убрано лишнее присваивание "model =" — fit меняет pipeline на месте

    auc = evaluate_model(pipeline, X_test, Y_test)

    print(auc)