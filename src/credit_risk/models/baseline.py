from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier

def build_baseline_pipeline(num_cols, cat_cols):
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
    ])

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', LogisticRegression(max_iter=1000)),
    ])

    return pipeline


def evaluate_model(pipeline, X_test, y_test):
    prediction = pipeline.predict_proba(X_test)[:, 1]  # ИЗМЕНЕНО: добавлен срез [:, 1] — берём только вероятность класса 1, а не обе колонки
    return roc_auc_score(y_test, prediction)
    

def build_lightgbm_pipeline(num_cols, cat_cols):
    preprocessor = ColumnTransformer([
        ('num', 'passthrough', num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
    ])

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', LGBMClassifier(n_estimators=200, random_state=42)),
    ])

    return pipeline

