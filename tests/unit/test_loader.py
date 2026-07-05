import pytest
from credit_risk.data.loader import load_german_credit


def test_load_german_credit_returns_dataframe():
    df = load_german_credit()
    assert not df.empty


def test_load_german_credit_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_german_credit("несуществующий_путь.csv")