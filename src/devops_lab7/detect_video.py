"""Завдання 8* (+ «Додатково»). Виділення облич у відеофайлі або з веб-камери.

Запуск:
    uv run lab7-video                    # data/video_faces.avi -> output/video_detected.avi
    uv run lab7-video --recognize        # додатково: підписати осіб з бази (LBPH)
    uv run lab7-video --show 1           # відтворювати відео у вікні під час обробки
    uv run lab7-video --webcam 0         # веб-камера (вихід — клавіша q або Esc)
"""

import argparse
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from devops_lab7.console import OUTPUT_DIR, section, setup_console, table
from devops_lab7.drawing import BLUE, GREEN, RED, caption, draw_boxes, grid, save
from devops_lab7.face_data import VIDEO_PATH, load_database
from devops_lab7.face_detectors import FaceDetector, HaarCascadeDetector, Image
from devops_lab7.face_recognizers import FaceRecognizer, LbphRecognizer

SAMPLE_FRAMES = (15, 55, 95, 135)
WINDOW = "Lab7 Task 8 - faces in video"


def process_frame(
    frame: Image,
    detector: FaceDetector,
    recognizer: FaceRecognizer | None,
    names: dict[int, str],
    counter: Counter[str],
) -> tuple[Image, int, float]:
    start = perf_counter()
    boxes = detector.detect(frame)
    elapsed = perf_counter() - start
    if recognizer is None:
        return draw_boxes(frame, boxes, BLUE), len(boxes), elapsed
    annotated = frame.copy()
    for box in boxes:
        prediction = recognizer.predict(box.crop(frame))
        name = names[prediction.label] if prediction.label is not None else "Unknown"
        counter[name] += 1
        color = GREEN if prediction.is_known else RED
        annotated = draw_boxes(annotated, [box], color, [f"{name} ({prediction.distance:.0f})"])
    return annotated, len(boxes), elapsed


def hud(frame: Image, n: int, total: int, faces: int, fps: float) -> None:
    text = f"frame {n + 1}/{total if total > 0 else '?'}  faces: {faces}  {fps:.0f} fps"
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 34), (255, 255, 255), -1)
    cv2.putText(frame, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2, cv2.LINE_AA)


def main(argv: Sequence[str] | None = None) -> None:
    setup_console()
    parser = argparse.ArgumentParser(prog="lab7-video", description="ЛР7, завдання 8")
    parser.add_argument("--input", type=Path, default=VIDEO_PATH)
    parser.add_argument("--webcam", type=int, default=None, help="номер веб-камери (0, 1 …)")
    parser.add_argument("--recognize", action="store_true", help="розпізнавати осіб з бази")
    parser.add_argument("--show", type=float, default=0, help="показувати кадри у вікні")
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args(argv)

    source: int | str = args.webcam if args.webcam is not None else str(args.input)
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise SystemExit(f"Не вдалося відкрити джерело {source}. Спершу: uv run lab7-prepare")
    fps = capture.get(cv2.CAP_PROP_FPS) or 20.0
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) if args.webcam is None else -1
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    detector = HaarCascadeDetector("default")
    recognizer: FaceRecognizer | None = None
    names: dict[int, str] = {}
    if args.recognize:
        faces, labels, names = load_database(detector)
        recognizer = LbphRecognizer()
        recognizer.train(faces, labels)

    suffix = "recognized" if args.recognize else "detected"
    out_path = OUTPUT_DIR / f"video_{suffix}.avi"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter.fourcc(*"MJPG"), fps, (width, height))

    if args.recognize:
        section("ЛР7. Додатково: розпізнавання осіб у відео (LBPH)")
    else:
        section("ЛР7, завдання 8. Виділення облич у відео")
    origin = f"веб-камера {args.webcam}" if args.webcam is not None else args.input.as_posix()
    frames_text = total if total > 0 else "—"
    print(
        f"Джерело: {origin} ({width}x{height}, {fps:.0f} fps, кадрів: {frames_text}); "
        f"детектор: {detector.name}"
    )

    counter: Counter[str] = Counter()
    per_frame: list[int] = []
    detect_time = 0.0
    samples: list[Image] = []
    started = perf_counter()
    n = 0
    while True:
        ok, raw = capture.read()
        if not ok or (args.max_frames and n >= args.max_frames):
            break
        frame = np.asarray(raw, dtype=np.uint8)
        annotated, count, elapsed = process_frame(frame, detector, recognizer, names, counter)
        per_frame.append(count)
        detect_time += elapsed
        hud(annotated, n, total, count, (n + 1) / max(perf_counter() - started, 1e-6))
        writer.write(annotated)
        if n in SAMPLE_FRAMES:
            samples.append(caption(annotated, f"frame {n + 1}: {count} face(s)"))
            save(OUTPUT_DIR / f"video_{suffix}_frame_{n + 1:03d}.png", annotated)
        if args.show or args.webcam is not None:
            cv2.imshow(WINDOW, annotated)
            delay = max(1, int(1000 / fps)) if args.webcam is None else 1
            if cv2.waitKey(delay) & 0xFF in (ord("q"), 27):
                break
        n += 1
    capture.release()
    writer.release()
    if args.show or args.webcam is not None:
        cv2.destroyAllWindows()

    elapsed_total = perf_counter() - started
    with_faces = sum(1 for c in per_frame if c)
    rows = [
        ["Оброблено кадрів", str(len(per_frame))],
        ["Кадрів з обличчями", f"{with_faces} ({with_faces / max(len(per_frame), 1):.0%})"],
        ["Усього рамок", str(sum(per_frame))],
        ["Облич у кадрі: мін / макс", f"{min(per_frame, default=0)} / {max(per_frame, default=0)}"],
        ["Середній час детекції кадру", f"{detect_time / max(len(per_frame), 1) * 1000:.1f} мс"],
        ["Швидкість обробки", f"{len(per_frame) / max(elapsed_total, 1e-6):.1f} кадр/с"],
    ]
    print(table(["Показник", "Значення"], rows))
    if counter:
        print("\nСкільки разів кожну особу підписано у відео:")
        print(table(["Особа", "Кадрів"], [[name, str(c)] for name, c in counter.most_common()]))
    if samples:
        save(OUTPUT_DIR / f"video_{suffix}_frames.png", grid(samples, columns=2))
    print(f"\nЗбережено: {out_path.as_posix()}, кадри output/video_{suffix}_frame_*.png")


if __name__ == "__main__":
    main()
