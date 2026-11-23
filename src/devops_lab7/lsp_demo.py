"""Завдання 10. Демонстрація принципу підстановки Лісков (LSP) у коді ЛР7.

Один і той самий клієнтський код викликається з об'єктами різних підкласів трьох ієрархій
(ClassifierModel, FaceDetector, FaceRecognizer) і працює коректно без перевірок типу.
Наприкінці показано, що сталося б, якби клієнт залежав від деталей підкласу
(«чужий» поріг відстані) — тобто якби LSP було порушено.

Запуск: uv run lab7-lsp
"""

from collections.abc import Sequence

from devops_lab7.console import section, setup_console, table
from devops_lab7.drawing import read_image
from devops_lab7.face_data import DB_DIR, PHOTO_PATH, load_database, load_truth
from devops_lab7.face_detectors import FaceDetector, HaarCascadeDetector, Image, all_detectors
from devops_lab7.face_recognizers import FaceRecognizer, LbphRecognizer, all_recognizers
from devops_lab7.ml_models import (
    ClassifierModel,
    RandomForestModel,
    SvmModel,
    evaluate,
    load_digits_split,
)


def accuracy_of(model: ClassifierModel) -> float:
    """Клієнт 1: знає лише базовий клас ClassifierModel."""
    return evaluate(model, load_digits_split(), repeats=1).accuracy


def faces_found(detector: FaceDetector, image: Image) -> int:
    """Клієнт 2: знає лише базовий клас FaceDetector."""
    return len(detector.detect(image))


def known_people(recognizer: FaceRecognizer, faces: Sequence[Image]) -> int:
    """Клієнт 3: знає лише базовий клас FaceRecognizer (поріг — справа підкласу)."""
    return sum(recognizer.predict(face).is_known for face in faces)


def main() -> None:
    setup_console()
    section("ЛР7, завдання 10. Принцип підстановки Лісков (LSP)")
    print("Клієнт 1 — accuracy_of(model: ClassifierModel):")
    rows = [
        [type(m).__name__, m.name, f"{accuracy_of(m):.4f}"]
        for m in (SvmModel(), RandomForestModel())
    ]
    print(table(["Клас", "Назва", "Accuracy"], rows))

    photo = read_image(PHOTO_PATH)
    if photo is None or not DB_DIR.exists():
        raise SystemExit("Немає даних. Спершу виконайте: uv run lab7-prepare")
    print("\nКлієнт 2 — faces_found(detector: FaceDetector, photo):")
    rows = [[type(d).__name__, d.name, str(faces_found(d, photo))] for d in all_detectors()]
    print(table(["Клас", "Назва", "Облич"], rows))

    detector = HaarCascadeDetector("default")
    faces, labels, _ = load_database(detector)
    truth = load_truth()
    crops = [box.crop(photo) for box in detector.detect(photo)]
    expected = sum(t.known for t in truth)
    print(
        f"\nКлієнт 3 — known_people(recognizer: FaceRecognizer, faces) "
        f"(на фото {len(crops)} облич, з них {expected} осіб з бази):"
    )
    rows = []
    for recognizer in all_recognizers():
        recognizer.train(faces, labels)
        rows.append(
            [
                type(recognizer).__name__,
                f"{recognizer.threshold:g}",
                str(known_people(recognizer, crops)),
            ]
        )
    print(table(["Клас", "Власний поріг", "Відомих осіб"], rows))

    section("Якби LSP було порушено: клієнт використовує поріг LBPH для всіх підкласів")
    lbph_threshold = LbphRecognizer.threshold
    rows = []
    for recognizer in all_recognizers():
        recognizer.train(faces, labels)
        accepted = sum(recognizer.predict(c).distance <= lbph_threshold for c in crops)
        rows.append([type(recognizer).__name__, f"{lbph_threshold:g}", str(accepted)])
    print(table(["Клас", "Поріг клієнта", "Відомих осіб"], rows))
    print(
        "Відстані Eigenfaces/Fisherfaces вимірюються в сотнях і тисячах, тож «чужий» поріг "
        "відкидає всіх.\nТому поріг інкапсульовано в підкласі, а predict() повертає вже "
        "готовий результат (label або None)."
    )


if __name__ == "__main__":
    main()
