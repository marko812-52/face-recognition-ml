"""Завдання 1–6. Моделі класифікації та їх оцінювання (scikit-learn, датасет Digits).

Принцип підстановки Лісков (LSP): функція evaluate() працює з будь-яким підкласом
ClassifierModel і не знає, SVM це чи випадковий ліс. Кожен підклас дотримується
контракту базового класу: fit() приймає ті самі дані й повертає self, predict()
повертає мітки тієї самої форми та з тієї самої множини класів, не змінює вхідні дані
і не кидає нових винятків. Тому SvmModel і RandomForestModel взаємозамінні.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from statistics import median
from time import perf_counter
from typing import Any, Final, Self

import numpy as np
from numpy.typing import NDArray
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

RANDOM_STATE: Final = 42
TEST_SIZE: Final = 0.25

Features = NDArray[np.float64]
Labels = NDArray[np.int64]


class NotFittedError(RuntimeError):
    """predict() викликано до fit() — однаково для всіх підкласів (частина контракту)."""


class ClassifierModel(ABC):
    """Базовий клас моделі: спільний контракт fit / predict для всіх алгоритмів."""

    name: str = "base"

    def __init__(self) -> None:
        self._estimator: Any = None
        self._classes: Labels | None = None

    @abstractmethod
    def _build(self) -> Any:
        """Створює незатренований estimator scikit-learn (деталь реалізації підкласу)."""

    def fit(self, features: Features, labels: Labels) -> Self:
        """Навчає модель. Передумова: len(features) == len(labels) > 0."""
        if len(features) != len(labels) or len(features) == 0:
            raise ValueError("features і labels мають бути непорожніми й однакової довжини")
        self._estimator = self._build()
        self._estimator.fit(features, labels)
        self._classes = np.unique(labels)
        return self

    def predict(self, features: Features) -> Labels:
        """Повертає мітку класу для кожного рядка features (лише з навчальних класів)."""
        if self._estimator is None:
            raise NotFittedError(f"модель {self.name} ще не навчена: спершу викличте fit()")
        return np.asarray(self._estimator.predict(features), dtype=np.int64)

    @property
    def classes(self) -> Labels:
        if self._classes is None:
            raise NotFittedError(f"модель {self.name} ще не навчена")
        return self._classes

    def describe(self) -> str:
        return f"{self.name}: {self._build()!r}"


class SvmModel(ClassifierModel):
    """Метод опорних векторів (SVC, RBF-ядро) зі стандартизацією ознак."""

    name = "SVM (SVC, RBF)"

    def __init__(self, c: float = 10.0, gamma: str | float = "scale") -> None:
        super().__init__()
        self.c = c
        self.gamma = gamma

    def _build(self) -> Any:
        return make_pipeline(StandardScaler(), SVC(kernel="rbf", C=self.c, gamma=self.gamma))


class RandomForestModel(ClassifierModel):
    """Випадковий ліс — ансамбль дерев рішень."""

    name = "Random Forest"

    def __init__(self, n_estimators: int = 300, random_state: int = RANDOM_STATE) -> None:
        super().__init__()
        self.n_estimators = n_estimators
        self.random_state = random_state

    def _build(self) -> Any:
        return RandomForestClassifier(
            n_estimators=self.n_estimators, random_state=self.random_state, n_jobs=-1
        )


@dataclass(frozen=True)
class DatasetSplit:
    """Навчальна і тестова вибірки."""

    x_train: Features
    x_test: Features
    y_train: Labels
    y_test: Labels
    images_test: NDArray[np.float64]
    target_names: list[str]


def load_digits_split(test_size: float = TEST_SIZE) -> DatasetSplit:
    """Завантажує Digits (1797 зображень 8×8) і стратифіковано ділить на train/test."""
    digits = load_digits()
    indices = np.arange(len(digits.target))
    x_train, x_test, y_train, y_test, _, idx_test = train_test_split(
        digits.data,
        digits.target,
        indices,
        test_size=test_size,
        stratify=digits.target,
        random_state=RANDOM_STATE,
    )
    return DatasetSplit(
        x_train=np.asarray(x_train, dtype=np.float64),
        x_test=np.asarray(x_test, dtype=np.float64),
        y_train=np.asarray(y_train, dtype=np.int64),
        y_test=np.asarray(y_test, dtype=np.int64),
        images_test=np.asarray(digits.images[idx_test], dtype=np.float64),
        target_names=[str(name) for name in digits.target_names],
    )


@dataclass(frozen=True)
class EvaluationResult:
    """Результати оцінювання однієї моделі."""

    model_name: str
    fit_seconds: float
    predict_seconds: float
    accuracy: float
    report_text: str
    report: dict[str, Any]
    confusion: NDArray[np.int64]
    predictions: Labels = field(repr=False)


def evaluate(model: ClassifierModel, split: DatasetSplit, repeats: int = 5) -> EvaluationResult:
    """Навчає й оцінює БУДЬ-ЯКИЙ ClassifierModel (LSP): час fit/predict — медіана повторів."""
    fit_times: list[float] = []
    predict_times: list[float] = []
    predictions: Labels | None = None
    for _ in range(repeats):
        start = perf_counter()
        model.fit(split.x_train, split.y_train)
        fit_times.append(perf_counter() - start)
        start = perf_counter()
        predictions = model.predict(split.x_test)
        predict_times.append(perf_counter() - start)
    assert predictions is not None
    return EvaluationResult(
        model_name=model.name,
        fit_seconds=median(fit_times),
        predict_seconds=median(predict_times),
        accuracy=float(accuracy_score(split.y_test, predictions)),
        report_text=str(
            classification_report(split.y_test, predictions, digits=3, zero_division=0)
        ),
        report=dict(
            classification_report(
                split.y_test, predictions, digits=3, output_dict=True, zero_division=0
            )
        ),
        confusion=np.asarray(confusion_matrix(split.y_test, predictions), dtype=np.int64),
        predictions=predictions,
    )
