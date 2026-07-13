from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression


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