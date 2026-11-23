"""Розпізнавачі облич OpenCV (модуль cv2.face) з єдиним контрактом — LSP.

Контракт FaceRecognizer:
* train(faces, labels) — приймає непорожній список облич будь-якого розміру та цілі мітки;
* predict(face) -> Prediction: label — одна з навчальних міток або None («невідома особа»),
  distance >= 0 (менше — краще).
Шкала відстані в кожного алгоритму своя (LBPH — десятки, Eigenfaces — тисячі), тому
поріг «невідомої особи» інкапсульовано в підкласі. Якби клієнт сам порівнював distance
з одним порогом, заміна LBPH на Eigenfaces зламала б програму — це було б порушенням LSP.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final

import cv2
import numpy as np
from numpy.typing import NDArray

Image = NDArray[np.uint8]
FACE_SIZE: Final = (100, 100)
MARGIN: Final = 0.1  # частка рамки, що відкидається з боків


@dataclass(frozen=True, slots=True)
class Prediction:
    label: int | None  # None — обличчя не схоже на жодну особу з бази
    distance: float

    @property
    def is_known(self) -> bool:
        return self.label is not None


def normalize_face(face: Image) -> Image:
    """Однакова підготовка для навчання і розпізнавання: сірий, обрізані краї рамки
    (тло й волосся), 100×100, вирівняна гістограма."""
    gray = face if face.ndim == 2 else cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    dx, dy = int(w * MARGIN), int(h * MARGIN)
    inner = gray[dy : h - dy // 2, dx : w - dx]
    resized = cv2.resize(inner, FACE_SIZE, interpolation=cv2.INTER_AREA)
    return np.asarray(cv2.equalizeHist(resized), dtype=np.uint8)


def augment(face: Image) -> list[Image]:
    """Розширення бази: дзеркальне відображення та повороти на ±6°."""
    h, w = face.shape[:2]
    variants = [face, np.asarray(cv2.flip(face, 1), dtype=np.uint8)]
    for angle in (-6, 6):
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        rotated = cv2.warpAffine(face, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)
        variants.append(np.asarray(rotated, dtype=np.uint8))
    return variants


class FaceRecognizer(ABC):
    """Базовий клас розпізнавача облич."""

    name: str = "base"
    threshold: float = float("inf")

    def __init__(self) -> None:
        self._model: Any = None
        self._labels: set[int] = set()

    @abstractmethod
    def _create(self) -> Any:
        """Створює модель cv2.face.* (деталь реалізації)."""

    def train(self, faces: Sequence[Image], labels: Sequence[int]) -> None:
        if not faces or len(faces) != len(labels):
            raise ValueError("потрібні непорожні faces і labels однакової довжини")
        self._model = self._create()
        prepared, expanded = [], []
        for face, label in zip(faces, labels, strict=True):
            for variant in augment(normalize_face(face)):
                prepared.append(variant)
                expanded.append(label)
        self._model.train(prepared, np.asarray(expanded, dtype=np.int32))
        self._labels = {int(label) for label in labels}

    def predict(self, face: Image) -> Prediction:
        if self._model is None:
            raise RuntimeError(f"{self.name}: спершу викличте train()")
        label, distance = self._model.predict(normalize_face(face))
        distance = max(0.0, float(distance))
        known = distance <= self.threshold and int(label) in self._labels
        return Prediction(int(label) if known else None, distance)


class LbphRecognizer(FaceRecognizer):
    """Local Binary Patterns Histograms — стійкий до освітлення, вчиться інкрементно."""

    name = "LBPH"
    threshold = 63.0

    def _create(self) -> Any:
        return cv2.face.LBPHFaceRecognizer_create(  # type: ignore[attr-defined]
            radius=2, neighbors=8, grid_x=7, grid_y=7
        )


class EigenfaceRecognizer(FaceRecognizer):
    """Eigenfaces — метод головних компонент (PCA) у просторі облич."""

    name = "Eigenfaces"
    threshold = 3300.0

    def _create(self) -> Any:
        return cv2.face.EigenFaceRecognizer_create(num_components=60)  # type: ignore[attr-defined]


class FisherfaceRecognizer(FaceRecognizer):
    """Fisherfaces — лінійний дискримінантний аналіз (LDA) після PCA."""

    name = "Fisherfaces"
    threshold = 710.0

    def _create(self) -> Any:
        return cv2.face.FisherFaceRecognizer_create()  # type: ignore[attr-defined]


def all_recognizers() -> list[FaceRecognizer]:
    return [LbphRecognizer(), EigenfaceRecognizer(), FisherfaceRecognizer()]
