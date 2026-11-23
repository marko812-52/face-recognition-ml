"""Завдання 7*. Детекція облич на фото з кількома особами кількома класифікаторами OpenCV.

Запуск:
    uv run lab7-detect                       # data/group_photo.png, усі детектори
    uv run lab7-detect --image моє_фото.jpg  # будь-яке інше фото
    uv run lab7-detect --show 8              # показати вікно з результатом 8 секунд
"""

import argparse
from collections.abc import Sequence
from pathlib import Path

import cv2

from devops_lab7.console import OUTPUT_DIR, section, setup_console, table
from devops_lab7.drawing import BLUE, caption, draw_boxes, grid, read_image, save, show
from devops_lab7.face_data import PHOTO_PATH, TRUTH_PATH, load_truth
from devops_lab7.face_detectors import FaceBox, FaceDetector, Image, all_detectors

IOU_MATCH = 0.3


def score(boxes: Sequence[FaceBox], truth: Sequence[FaceBox]) -> tuple[int, int, int]:
    """TP, FP, FN: рамка вважається правильною, якщо IoU з еталонною >= 0.3."""
    matched: set[int] = set()
    tp = 0
    for box in boxes:
        best = max(range(len(truth)), key=lambda i: box.iou(truth[i]), default=None)
        if best is not None and best not in matched and box.iou(truth[best]) >= IOU_MATCH:
            matched.add(best)
            tp += 1
    return tp, len(boxes) - tp, len(truth) - tp


def slug(detector: FaceDetector) -> str:
    return detector.name.lower().replace(" + ", "_").replace(" ", "_")


def run_detector(
    detector: FaceDetector, image: Image, truth: list[FaceBox]
) -> tuple[list[str], Image]:
    """Клієнтський код: однаковий для будь-якого підкласу FaceDetector (LSP)."""
    boxes, seconds = detector.timed_detect(image)
    annotated = draw_boxes(image, boxes, BLUE)
    save(OUTPUT_DIR / f"detect_{slug(detector)}.png", annotated)
    row = [detector.name, str(len(boxes))]
    if truth:
        tp, fp, fn = score(boxes, truth)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        row += [str(tp), str(fp), str(fn), f"{precision:.2f}", f"{recall:.2f}"]
    row.append(f"{seconds * 1000:.0f}")
    return row, caption(annotated, f"{detector.name}: {len(boxes)} faces")


def main(argv: Sequence[str] | None = None) -> None:
    setup_console()
    parser = argparse.ArgumentParser(prog="lab7-detect", description="ЛР7, завдання 7")
    parser.add_argument("--image", type=Path, default=PHOTO_PATH)
    parser.add_argument("--show", type=float, default=0, help="показати вікно N секунд")
    args = parser.parse_args(argv)

    image = read_image(args.image)
    if image is None:
        raise SystemExit(f"Не вдалося відкрити {args.image}. Спершу виконайте: uv run lab7-prepare")
    truth = []
    if args.image == PHOTO_PATH and TRUTH_PATH.exists():
        truth = [FaceBox(*t.box) for t in load_truth()]

    section("ЛР7, завдання 7. Детекція облич на фото (OpenCV, каскади Хаара)")
    print(
        f"Фото: {args.image.as_posix()} ({image.shape[1]}x{image.shape[0]}), "
        f"осіб на фото (еталон): {len(truth) or 'невідомо'}"
    )
    print(f"OpenCV {cv2.__version__}; scaleFactor=1.05, minNeighbors=5, minSize=40x40\n")

    rows, tiles = [], []
    for detector in all_detectors():
        row, tile = run_detector(detector, image, truth)
        rows.append(row)
        tiles.append(tile)
    header = ["Детектор", "Знайдено"]
    if truth:
        header += ["TP", "FP", "FN", "Precision", "Recall"]
    print(table([*header, "Час, мс"], rows))

    save(OUTPUT_DIR / "detect_all.png", grid(tiles, columns=2))
    print("\nЗбережено: output/detect_<детектор>.png (рамки), output/detect_all.png (порівняння)")
    main_result = draw_boxes(image, all_detectors()[0].detect(image), BLUE)
    show("Lab7 Task 7 - face detection (Haar default)", main_result, args.show)


if __name__ == "__main__":
    main()
