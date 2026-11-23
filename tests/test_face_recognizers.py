"""Контрактні тести FaceRecognizer для LBPH / Eigenfaces / Fisherfaces (LSP)."""

import numpy as np
import pytest

from devops_lab7.face_data import load_olivetti, subject_image
from devops_lab7.face_recognizers import (
    FACE_SIZE,
    FaceRecognizer,
    Prediction,
    all_recognizers,
    augment,
    normalize_face,
)

RECOGNIZER_TYPES = [type(r) for r in all_recognizers()]


@pytest.fixture(params=RECOGNIZER_TYPES, ids=lambda cls: cls.__name__)
def trained(request: pytest.FixtureRequest) -> FaceRecognizer:
    images, target = load_olivetti()
    faces = [subject_image(images, target, s, i) for s in (1, 7, 9) for i in range(6)]
    labels = [label for label in (0, 1, 2) for _ in range(6)]
    recognizer: FaceRecognizer = request.param()
    recognizer.train(faces, labels)
    return recognizer


@pytest.mark.parametrize("cls", RECOGNIZER_TYPES, ids=lambda cls: cls.__name__)
def test_predict_before_train_raises(cls: type[FaceRecognizer]) -> None:
    with pytest.raises(RuntimeError):
        cls().predict(np.zeros((64, 64), np.uint8))


def test_prediction_contract(trained: FaceRecognizer) -> None:
    images, target = load_olivetti()
    for subject, label in ((1, 0), (7, 1), (9, 2)):
        prediction = trained.predict(subject_image(images, target, subject, 0))  # знімок з бази
        assert isinstance(prediction, Prediction)
        assert prediction.distance >= 0
        assert prediction.label == label


def test_label_is_known_or_none(trained: FaceRecognizer) -> None:
    images, target = load_olivetti()
    prediction = trained.predict(subject_image(images, target, 38, 0))  # особи немає в базі
    assert prediction.label in (0, 1, 2, None)


@pytest.mark.parametrize("cls", RECOGNIZER_TYPES, ids=lambda cls: cls.__name__)
def test_train_rejects_bad_input(cls: type[FaceRecognizer]) -> None:
    with pytest.raises(ValueError):
        cls().train([], [])


def test_normalize_and_augment() -> None:
    face = np.random.default_rng(0).integers(0, 255, (180, 150, 3), dtype=np.uint8)
    normalized = normalize_face(face)
    assert normalized.shape == FACE_SIZE and normalized.dtype == np.uint8
    assert len(augment(normalized)) == 4
