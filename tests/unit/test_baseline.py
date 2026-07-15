import pytest
from credit_risk.data.loader import load_german_credit
from credit_risk.models.baseline import build_baseline_pipeline
from credit_risk.features.prepare import split_features_target


def test_build_baseline_pipeline():
    df = load_german_credit()
    x, y = split_features_target(df)
    num_cols = x.select_dtypes(include='number').columns.tolist()
    cat_cols = x.select_dtypes(include='object').columns.tolist()

    pipeline = build_baseline_pipeline(num_cols, cat_cols)

    pipeline.fit(x, y)


    predict = pipeline.predict(x).shape
    assert predict[0] == x.shape[0] #для теста не кртитчно обучение и предикт на x. Цель лишь удостоверится что всё сработало
