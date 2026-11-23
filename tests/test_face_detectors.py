"""Контрактні тести FaceDetector для всіх підкласів (LSP) та FaceBox."""

import numpy as np
import pytest

from devops_lab7.face_data import background, load_olivetti, paste_face, subject_image
from devops_lab7.face_detectors import (
    EyeVerifiedDetector,
    FaceBox,
    FaceDetector,
    HaarCascadeDetector,
    all_detectors,
)

DETECTORS = all_detectors()


@pytest.fixture(scope="module")
def portrait() -> np.ndarray:
    images, target = load_olivetti()
    canvas = background(320, 360, seed=3)
    paste_face(canvas, subject_image(images, target, 7, 0), 160, 150, 2.6)
    return canvas


@pytest.mark.parametrize("detector", DETECTORS, ids=lambda d: d.name)
def test_empty_scene_returns_empty_list(detector: FaceDetector) -> None:
    assert detector.detect(background(400, 300, seed=5)) == []
    assert detector.detect(np.zeros((0, 0, 3), np.uint8)) == []


@pytest.mark.parametrize("detector", DETECTORS, ids=lambda d: d.name)
def test_boxes_inside_image_and_input_unchanged(
    detector: FaceDetector, portrait: np.ndarray
) -> None:
    original = portrait.copy()
    for image in (portrait, portrait[:, :, 0]):  # BGR і півтонове
        for box in detector.detect(image):
            assert box.x >= 0 and box.y >= 0 and box.w > 0 and box.h > 0
            assert box.x + box.w <= image.shape[1] and box.y + box.h <= image.shape[0]
    np.testing.assert_array_equal(portrait, original)


@pytest.mark.parametrize("cascade", ["default", "alt", "alt2"])
def test_frontal_cascades_find_single_face(cascade: str, portrait: np.ndarray) -> None:
    boxes = HaarCascadeDetector(cascade).detect(portrait)
    assert len(boxes) == 1
    assert boxes[0].iou(FaceBox(77, 67, 166, 166)) > 0.4


def test_eye_verified_detector_is_subset_of_inner(portrait: np.ndarray) -> None:
    inner = HaarCascadeDetector("default")
    assert set(EyeVerifiedDetector(inner).detect(portrait)) <= set(inner.detect(portrait))


def test_unknown_cascade_is_rejected() -> None:
    with pytest.raises(ValueError):
        HaarCascadeDetector("no_such_cascade")


def test_iou() -> None:
    a = FaceBox(0, 0, 10, 10)
    assert a.iou(a) == 1.0
    assert a.iou(FaceBox(20, 20, 5, 5)) == 0.0
    assert a.iou(FaceBox(5, 0, 10, 10)) == pytest.approx(50 / 150)
