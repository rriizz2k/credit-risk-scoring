from credit_risk.data.loader import load_german_credit
from credit_risk.models.baseline import build_baseline_pipeline, evaluate_model, build_lightgbm_pipeline, evaluate_model_detailed
from credit_risk.features.prepare import split_features_target, add_engineered_features
from sklearn.model_selection import train_test_split
import mlflow

if __name__ == "__main__":
    mlflow.set_experiment("credit-risk-baseline")

    
    df = load_german_credit()
    df = add_engineered_features(df)
    x, y = split_features_target(df)
    X_train, X_test, Y_train, Y_test = train_test_split(
    x, y, test_size=0.2, stratify=y, random_state=42
    )  # ИЗМЕНЕНО: раньше вызывалось без аргументов — теперь передаём x, y и параметры разбиения



    num_cols = X_train.select_dtypes(include='number').columns.tolist()  # ИЗМЕНЕНО: было x.select_dtypes(...) — теперь именно X_train, а не весь x
    cat_cols = X_train.select_dtypes(include='object').columns.tolist()  # аналогично

    print('-'*20)
    print('num',num_cols)
    print('-'*20)
    print('cat',cat_cols)
    print('-'*20)

    num_cols.remove('credit_amount_per_month')
    num_cols.remove('credit_amount_log')
    #num_cols.remove('credit_amount')
    #num_cols.remove('duration')

    msg = 'without new features + balanced + ranstate42'

    with mlflow.start_run():
        pipeline = build_baseline_pipeline(num_cols, cat_cols)
        pipeline.fit(X_train, Y_train)  # ИЗМЕНЕНО: убрано лишнее присваивание "model =" — fit меняет pipeline на месте
        auc = evaluate_model(pipeline, X_test, Y_test)
        #print(auc)
        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_metric("roc_auc", auc)
        mlflow.log_metric('recall', evaluate_model_detailed(pipeline, X_test, Y_test)['1']['recall'])
        mlflow.log_param("Description", msg)

    with mlflow.start_run():
        pipeline = build_lightgbm_pipeline(num_cols, cat_cols)
        pipeline.fit(X_train, Y_train)  # ИЗМЕНЕНО: убрано лишнее присваивание "model =" — fit меняет pipeline на месте
        auc = evaluate_model(pipeline, X_test, Y_test)
        #print(auc)
        mlflow.log_param("model", "LightGBM")
        mlflow.log_metric("roc_auc", auc)
        mlflow.log_metric('recall', evaluate_model_detailed(pipeline, X_test, Y_test)['1']['recall'])
        mlflow.log_param("Description", msg)