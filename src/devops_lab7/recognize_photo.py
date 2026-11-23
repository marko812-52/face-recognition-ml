"""Завдання 9. Розпізнавання конкретних осіб на груповому фото (база — підготовлені зображення).

Запуск:
    uv run lab7-recognize            # LBPH + порівняння з Eigenfaces / Fisherfaces
    uv run lab7-recognize --show 8   # показати вікно з підписаними рамками 8 секунд
"""

import argparse
from collections.abc import Sequence
from dataclasses import dataclass

from devops_lab7.console import OUTPUT_DIR, section, setup_console, table
from devops_lab7.drawing import GREEN, RED, caption, draw_boxes, grid, read_image, save, show
from devops_lab7.face_data import (
    DB_DIR,
    KNOWN_SUBJECTS,
    PHOTO_PATH,
    TEST_IMAGES,
    UNKNOWN_SUBJECTS,
    extract_face,
    load_database,
    load_olivetti,
    load_truth,
    person_label,
    subject_image,
)
from devops_lab7.face_detectors import FaceBox, FaceDetector, HaarCascadeDetector, Image
from devops_lab7.face_recognizers import FaceRecognizer, Prediction, all_recognizers

UNKNOWN = "Unknown"


@dataclass(frozen=True)
class Recognized:
    box: FaceBox
    prediction: Prediction
    name: str


def recognize(
    image: Image, detector: FaceDetector, recognizer: FaceRecognizer, names: dict[int, str]
) -> list[Recognized]:
    """Клієнтський код: працює з будь-якою парою підкласів FaceDetector і FaceRecognizer."""
    found = []
    for box in detector.detect(image):
        prediction = recognizer.predict(box.crop(image))
        name = names[prediction.label] if prediction.label is not None else UNKNOWN
        found.append(Recognized(box, prediction, name))
    return found


def annotate(image: Image, found: Sequence[Recognized]) -> Image:
    canvas = image.copy()
    for item in found:
        color = GREEN if item.prediction.is_known else RED
        canvas = draw_boxes(
            canvas, [item.box], color, [f"{item.name} ({item.prediction.distance:.0f})"]
        )
    return canvas


def holdout_scores(
    recognizer: FaceRecognizer, detector: FaceDetector, names: dict[int, str]
) -> tuple[int, int, int, int]:
    """Перевірка на знімках, яких немає в базі: (правильні, усього, відхилені невідомі, усього)."""
    images, target = load_olivetti()
    by_name = {name: label for label, name in names.items()}
    correct = total = 0
    for subject in KNOWN_SUBJECTS:
        for index in TEST_IMAGES:
            face = extract_face(subject_image(images, target, subject, index), detector)
            correct += recognizer.predict(face).label == by_name[person_label(subject)]
            total += 1
    rejected = unknown_total = 0
    for subject in UNKNOWN_SUBJECTS:
        for index in range(10):
            face = extract_face(subject_image(images, target, subject, index), detector)
            rejected += recognizer.predict(face).label is None
            unknown_total += 1
    return correct, total, rejected, unknown_total


def main(argv: Sequence[str] | None = None) -> None:
    setup_console()
    parser = argparse.ArgumentParser(prog="lab7-recognize", description="ЛР7, завдання 9")
    parser.add_argument("--show", type=float, default=0, help="показати вікно N секунд")
    args = parser.parse_args(argv)

    image = read_image(PHOTO_PATH)
    if image is None or not DB_DIR.exists():
        raise SystemExit("Немає даних. Спершу виконайте: uv run lab7-prepare")
    truth = load_truth()
    detector = HaarCascadeDetector("default")
    faces, labels, names = load_database(detector)

    section("ЛР7, завдання 9. Розпізнавання облич на груповому фото")
    print(
        f"База: {DB_DIR.as_posix()} — {len(names)} осіб x {len(faces) // len(names)} знімків = "
        f"{len(faces)} зображень (+ аугментація x4 під час навчання)"
    )
    print(f"Особи в базі: {', '.join(names.values())}")
    print(
        f"Фото: {PHOTO_PATH.as_posix()}; детектор: {detector.name}; "
        f"еталон: {sum(t.known for t in truth)} осіб з бази + "
        f"{sum(not t.known for t in truth)} невідомі\n"
    )

    summary, tiles = [], []
    for recognizer in all_recognizers():
        recognizer.train(faces, labels)
        found = recognize(image, detector, recognizer, names)
        correct = 0
        rows = []
        for item in found:
            expected = max(truth, key=lambda t: item.box.iou(FaceBox(*t.box)))
            want = expected.label if expected.known else UNKNOWN
            ok = item.name == want
            correct += ok
            rows.append(
                [
                    f"({item.box.x},{item.box.y}) {item.box.w}x{item.box.h}",
                    want,
                    item.name,
                    f"{item.prediction.distance:.1f}",
                    "так" if ok else "НІ",
                ]
            )
        held_ok, held_total, rej, unk_total = holdout_scores(recognizer, detector, names)
        summary.append(
            [
                recognizer.name,
                f"{recognizer.threshold:g}",
                f"{correct}/{len(found)}",
                f"{held_ok}/{held_total}",
                f"{rej}/{unk_total}",
            ]
        )
        annotated = annotate(image, found)
        save(OUTPUT_DIR / f"recognized_{recognizer.name.lower()}.png", annotated)
        tiles.append(caption(annotated, f"{recognizer.name}: {correct}/{len(found)} correct"))
        if recognizer.name == "LBPH":
            print(f"Результат {recognizer.name} (поріг відстані {recognizer.threshold:g}):")
            print(
                table(["Рамка (x,y) WxH", "Хто насправді", "Розпізнано", "Відстань", "Вірно"], rows)
            )
            print()

    section("Порівняння розпізнавачів (однаковий клієнтський код — LSP)")
    print(
        table(
            ["Метод", "Поріг", "Фото: вірно", "Інші знімки: вірно", "Невідомі відхилено"], summary
        )
    )
    save(OUTPUT_DIR / "recognized_all.png", grid(tiles, columns=1, width=900))
    print(
        "\nЗбережено: output/recognized_lbph.png, recognized_eigenfaces.png, "
        "recognized_fisherfaces.png, recognized_all.png"
    )
    lbph = read_image(OUTPUT_DIR / "recognized_lbph.png")
    if lbph is not None:
        show("Lab7 Task 9 - face recognition (LBPH)", lbph, args.show)


if __name__ == "__main__":
    main()
