"""Підготовка даних для завдань 7–9 (виконується один раз).

1. Завантажує датасет облич Olivetti (AT&T, 40 осіб × 10 знімків) через scikit-learn
   і зберігає його в data/olivetti_faces.npz.
2. Створює базу розпізнавання data/faces_db/<особа>/<n>.png (8 осіб × 8 знімків).
3. Складає групове фото data/group_photo.png (+ еталонні рамки group_photo_truth.json).
4. Створює відеофайл data/video_faces.avi.

Запуск: uv run lab7-prepare
"""

import argparse
import sys
from collections.abc import Sequence

import cv2
import numpy as np

from devops_lab7.console import section
from devops_lab7.face_data import (
    DB_DIR,
    NPZ_PATH,
    PHOTO_PATH,
    TRUTH_PATH,
    VIDEO_PATH,
    build_database,
    compose_group_photo,
    load_olivetti,
    render_video,
    save_truth,
)


def fetch() -> None:
    """Завантаження Olivetti (потрібен інтернет лише під час першого запуску)."""
    from sklearn.datasets import fetch_olivetti_faces

    bunch = fetch_olivetti_faces()
    faces = (np.asarray(bunch.images) * 255).round().astype(np.uint8)
    NPZ_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(NPZ_PATH, images=faces, target=np.asarray(bunch.target, dtype=np.int16))


def main(argv: Sequence[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(prog="lab7-prepare", description="ЛР7: підготовка даних")
    parser.add_argument("--fetch-only", action="store_true")
    args = parser.parse_args(argv)

    section("ЛР7. Підготовка даних для завдань 7–9")
    if not NPZ_PATH.exists():
        fetch()
    images, target = load_olivetti()
    print(
        f"Olivetti / AT&T: {len(images)} знімків {images.shape[1]}x{images.shape[2]}, "
        f"{len(np.unique(target))} осіб -> {NPZ_PATH.as_posix()}"
    )
    if args.fetch_only:
        return
    count = build_database(images, target)
    print(f"База облич: {DB_DIR.as_posix()}/<особа>/<n>.png — {count} зображень")
    photo, truth = compose_group_photo(images, target)
    cv2.imwrite(str(PHOTO_PATH), photo)
    save_truth(truth)
    known = sum(t.known for t in truth)
    print(
        f"Групове фото: {PHOTO_PATH.as_posix()} ({photo.shape[1]}x{photo.shape[0]}), "
        f"{len(truth)} осіб: {known} з бази, {len(truth) - known} невідомі; "
        f"еталон: {TRUTH_PATH.as_posix()}"
    )
    frames = render_video(images, target)
    print(f"Відеофайл: {VIDEO_PATH.as_posix()} (960x540, 20 fps, {frames} кадрів, MJPG)")


if __name__ == "__main__":
    main()
