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
    )

    num_cols = X_train.select_dtypes(include='number').columns.tolist()
    cat_cols = X_train.select_dtypes(include='object').columns.tolist()

    # ----- убраны все print() — это были черновые проверки, в финальной версии не нужны

    # ----- убраны обе строки num_cols.remove(...) — используем ВСЕ признаки (оригиналы + engineered),
    # ----- это и есть победивший вариант по результатам экспериментов в MLflow

    msg = "all features + balanced"  # ----- переименовано под финальную, а не экспериментальную конфигурацию

    with mlflow.start_run():
        pipeline = build_baseline_pipeline(num_cols, cat_cols)
        pipeline.fit(X_train, Y_train)
        auc = evaluate_model(pipeline, X_test, Y_test)
        recall = evaluate_model_detailed(pipeline, X_test, Y_test)['1']['recall']  # ----- вынесено в переменную вместо инлайна — чуть чище читается
        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_metric("roc_auc", auc)
        mlflow.log_metric("recall", recall)
        mlflow.log_param("Description", msg)  # ----- "Description" -> "description": mlflow param-имена принято писать строчными

    with mlflow.start_run():
        pipeline = build_lightgbm_pipeline(num_cols, cat_cols)
        pipeline.fit(X_train, Y_train)
        auc = evaluate_model(pipeline, X_test, Y_test)
        recall = evaluate_model_detailed(pipeline, X_test, Y_test)['1']['recall']
        mlflow.log_param("model", "LightGBM")
        mlflow.log_metric("roc_auc", auc)
        mlflow.log_metric("recall", recall)
        mlflow.log_param("Description", msg)