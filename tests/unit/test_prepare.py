import pytest
from credit_risk.features.prepare import split_features_target, add_engineered_features
from credit_risk.data.loader import load_german_credit


def test_split_features_target():
    df = load_german_credit()
    x, y = split_features_target(df)
    assert (
        ('class' not in x.columns.to_list())
        and (set(y.unique()) == {0, 1})
        and df.shape[0] == x.shape[0]
        and x.shape[0] == y.shape[0]
        and x.shape[1] > 1
    )


def test_split_features_target_bad_is_1():
    df = load_german_credit()
    bad_index = df[df['class'] == 'bad'].index[0]

    x, y = split_features_target(df)

    assert y.loc[bad_index] == 1

def test_add_engineered_features():
    df = load_german_credit()
    dfa = df['credit_amount'].copy(deep=True)
    df = add_engineered_features(df)
    assert 'credit_amount_log' in df.columns.to_list()
    assert dfa.equals(df['credit_amount'])
    assert 'credit_amount_per_month' in df.columns.to_list()