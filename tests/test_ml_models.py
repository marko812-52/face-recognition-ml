"""Контрактні тести ClassifierModel: ОДНАКОВІ для всіх підкласів — перевірка LSP."""

import numpy as np
import pytest

from devops_lab7.ml_models import (
    ClassifierModel,
    DatasetSplit,
    NotFittedError,
    RandomForestModel,
    SvmModel,
    evaluate,
)

MODELS = [SvmModel, RandomForestModel]


@pytest.fixture(params=MODELS, ids=lambda cls: cls.__name__)
def model(request: pytest.FixtureRequest) -> ClassifierModel:
    instance: ClassifierModel = request.param()
    return instance


def test_split_is_stratified_75_25(digits: DatasetSplit) -> None:
    assert (len(digits.y_train), len(digits.y_test)) == (1347, 450)
    assert set(np.unique(digits.y_test)) == set(range(10))
    assert digits.x_train.shape[1] == 64


def test_predict_before_fit_raises_same_error(model: ClassifierModel, digits: DatasetSplit) -> None:
    with pytest.raises(NotFittedError):
        model.predict(digits.x_test)


def test_fit_returns_self(model: ClassifierModel, digits: DatasetSplit) -> None:
    assert model.fit(digits.x_train, digits.y_train) is model


def test_predict_contract(model: ClassifierModel, digits: DatasetSplit) -> None:
    x_copy = digits.x_test.copy()
    predictions = model.fit(digits.x_train, digits.y_train).predict(digits.x_test)
    assert predictions.shape == digits.y_test.shape
    assert predictions.dtype == np.int64
    assert set(np.unique(predictions)) <= set(model.classes.tolist())
    np.testing.assert_array_equal(digits.x_test, x_copy)  # вхідні дані не змінено


def test_fit_rejects_mismatched_lengths(model: ClassifierModel, digits: DatasetSplit) -> None:
    with pytest.raises(ValueError):
        model.fit(digits.x_train, digits.y_train[:-1])


def test_evaluate_accepts_any_model(model: ClassifierModel, digits: DatasetSplit) -> None:
    result = evaluate(model, digits, repeats=1)
    assert result.accuracy > 0.95
    assert result.confusion.shape == (10, 10)
    assert result.confusion.sum() == len(digits.y_test)
    assert result.fit_seconds > 0 and result.predict_seconds > 0
    assert {"precision", "recall", "f1-score"} <= set(result.report["macro avg"])
