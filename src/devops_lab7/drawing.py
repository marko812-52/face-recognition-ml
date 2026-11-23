"""Малювання рамок і підписів, показ вікон OpenCV, збереження зображень."""

from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from devops_lab7.face_detectors import FaceBox

Image = NDArray[np.uint8]
GREEN = (40, 170, 40)
RED = (40, 40, 220)
BLUE = (200, 110, 20)


def put_label(image: Image, text: str, box: FaceBox, color: tuple[int, int, int]) -> None:
    """Підпис над рамкою на заливці кольору рамки (латиницею — шрифти Hershey)."""
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
    (tw, th), base = cv2.getTextSize(text, font, scale, thick)
    y = box.y - 6 if box.y - th - 10 > 0 else box.y + box.h + th + 8
    cv2.rectangle(image, (box.x, y - th - 6), (box.x + tw + 8, y + base - 2), color, -1)
    cv2.putText(image, text, (box.x + 4, y - 3), font, scale, (255, 255, 255), thick, cv2.LINE_AA)


def draw_boxes(
    image: Image,
    boxes: Sequence[FaceBox],
    color: tuple[int, int, int] = GREEN,
    labels: Sequence[str] | None = None,
) -> Image:
    """Повертає копію зображення з рамками (вхідне зображення не змінюється)."""
    canvas = np.asarray(
        image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR),
        dtype=np.uint8,
    )
    for n, box in enumerate(boxes):
        cv2.rectangle(canvas, (box.x, box.y), (box.x + box.w, box.y + box.h), color, 3)
        if labels is not None:
            put_label(canvas, labels[n], box, color)
    return canvas


def caption(image: Image, text: str) -> Image:
    """Додає смугу із заголовком над зображенням."""
    bar = np.full((44, image.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (30, 30, 30), 2, cv2.LINE_AA)
    return np.asarray(np.vstack([bar, image]), dtype=np.uint8)


def grid(images: Sequence[Image], columns: int, width: int = 640) -> Image:
    """Складає зображення в сітку (для порівняння результатів кількох детекторів)."""
    tiles = []
    for img in images:
        scale = width / img.shape[1]
        tiles.append(
            cv2.resize(img, (width, int(img.shape[0] * scale)), interpolation=cv2.INTER_AREA)
        )
    height = max(t.shape[0] for t in tiles)
    blank = np.full((height, width, 3), 255, np.uint8)
    tiles = [
        cv2.copyMakeBorder(
            t, 0, height - t.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255)
        )
        for t in tiles
    ]
    while len(tiles) % columns:
        tiles.append(blank)
    rows = [np.hstack(tiles[i : i + columns]) for i in range(0, len(tiles), columns)]
    return np.asarray(np.vstack(rows), dtype=np.uint8)


def read_image(path: Path) -> Image | None:
    """cv2.imread із типом результату uint8 (None, якщо файл не вдалося прочитати)."""
    image = cv2.imread(str(path))
    return None if image is None else np.asarray(image, dtype=np.uint8)


def save(path: Path, image: Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def show(title: str, image: Image, seconds: float) -> None:
    """Показує вікно OpenCV на seconds секунд (0 — не показувати)."""
    if seconds <= 0:
        return
    cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(title, image)
    cv2.waitKey(int(seconds * 1000))
    cv2.destroyWindow(title)
