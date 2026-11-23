"""Детектори облич (OpenCV) з єдиним контрактом — демонстрація принципу Лісков.

Контракт FaceDetector.detect(image):
* приймає BGR- або півтонове зображення uint8 будь-якого розміру;
* повертає список FaceBox (можливо, порожній), кожна рамка лежить у межах зображення;
* не змінює вхідне зображення і не кидає винятків, якщо облич немає.
Клієнтський код (detect_photo, detect_video, recognize_photo) працює з будь-яким
підкласом, тому детектори можна замінювати без змін у клієнті.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from time import perf_counter
from typing import Final

import cv2
import numpy as np
from numpy.typing import NDArray

Image = NDArray[np.uint8]

HAAR_CASCADES: Final = {
    "default": "haarcascade_frontalface_default.xml",
    "alt": "haarcascade_frontalface_alt.xml",
    "alt2": "haarcascade_frontalface_alt2.xml",
    "alt_tree": "haarcascade_frontalface_alt_tree.xml",
}


@dataclass(frozen=True, slots=True)
class FaceBox:
    """Прямокутник обличчя: верхній лівий кут (x, y), ширина і висота."""

    x: int
    y: int
    w: int
    h: int

    @property
    def area(self) -> int:
        return self.w * self.h

    def iou(self, other: "FaceBox") -> float:
        """Intersection over Union — міра збігу двох рамок (0..1)."""
        x1, y1 = max(self.x, other.x), max(self.y, other.y)
        x2 = min(self.x + self.w, other.x + other.w)
        y2 = min(self.y + self.h, other.y + other.h)
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        union = self.area + other.area - inter
        return inter / union if union else 0.0

    def crop(self, image: Image) -> Image:
        return image[self.y : self.y + self.h, self.x : self.x + self.w]


def to_gray(image: Image) -> Image:
    """Півтонова копія зображення (вхідне зображення не змінюється)."""
    if image.ndim == 2:
        return image.copy()
    return np.asarray(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), dtype=np.uint8)


class FaceDetector(ABC):
    """Базовий клас детектора облич."""

    name: str = "base"

    @abstractmethod
    def _detect_gray(self, gray: Image) -> list[FaceBox]:
        """Пошук облич на півтоновому зображенні (реалізує підклас)."""

    def detect(self, image: Image) -> list[FaceBox]:
        """Шаблонний метод: однакова підготовка і постобробка для всіх підкласів."""
        if image.size == 0:
            return []
        gray = to_gray(image)
        height, width = gray.shape[:2]
        boxes = []
        for box in self._detect_gray(np.asarray(gray, dtype=np.uint8)):
            x, y = max(0, box.x), max(0, box.y)
            w, h = min(box.w, width - x), min(box.h, height - y)
            if w > 0 and h > 0:
                boxes.append(FaceBox(x, y, w, h))
        return sorted(boxes, key=lambda b: (b.y // 50, b.x))

    def timed_detect(self, image: Image) -> tuple[list[FaceBox], float]:
        start = perf_counter()
        boxes = self.detect(image)
        return boxes, perf_counter() - start


class HaarCascadeDetector(FaceDetector):
    """Каскадний класифікатор ознак Хаара (алгоритм Віоли–Джонса)."""

    def __init__(
        self,
        cascade: str = "default",
        scale_factor: float = 1.05,
        min_neighbors: int = 5,
        min_size: int = 40,
    ) -> None:
        if cascade not in HAAR_CASCADES:
            raise ValueError(f"невідомий каскад {cascade!r}, доступні: {', '.join(HAAR_CASCADES)}")
        path = cv2.data.haarcascades + HAAR_CASCADES[cascade]  # type: ignore[attr-defined]
        self._classifier = cv2.CascadeClassifier(path)
        if self._classifier.empty():
            raise RuntimeError(f"не вдалося завантажити каскад {path}")
        self.name = f"Haar {cascade}"
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size

    def _detect_gray(self, gray: Image) -> list[FaceBox]:
        found = self._classifier.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=(self.min_size, self.min_size),
        )
        return [FaceBox(int(x), int(y), int(w), int(h)) for (x, y, w, h) in found]


class EyeVerifiedDetector(FaceDetector):
    """Композиція: обличчя від іншого детектора залишаємо, лише якщо в ньому знайдено око.

    Підклас посилює постумову (менше хибних спрацювань), але не звужує передумову
    і повертає той самий тип результату — тому підстановка коректна.
    """

    def __init__(self, inner: FaceDetector, min_eyes: int = 1) -> None:
        self._inner = inner
        self._eyes = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye.xml"  # type: ignore[attr-defined]
        )
        self.min_eyes = min_eyes
        self.name = f"{inner.name} + eyes"

    def _detect_gray(self, gray: Image) -> list[FaceBox]:
        verified = []
        for box in self._inner.detect(gray):
            upper_half = gray[box.y : box.y + box.h // 2 + box.h // 8, box.x : box.x + box.w]
            eyes = self._eyes.detectMultiScale(
                upper_half, scaleFactor=1.1, minNeighbors=3, minSize=(box.w // 8, box.w // 8)
            )
            if len(eyes) >= self.min_eyes:
                verified.append(box)
        return verified


def all_detectors() -> list[FaceDetector]:
    """Набір детекторів для завдання 7 (порівняння кількох класифікаторів)."""
    detectors: list[FaceDetector] = [HaarCascadeDetector(name) for name in HAAR_CASCADES]
    detectors.append(EyeVerifiedDetector(HaarCascadeDetector("default")))
    return detectors
