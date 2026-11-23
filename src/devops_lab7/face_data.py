"""Підготовлені дані для завдань 7–9: база облич, групове фото та відеофайл.

Джерело облич — відкритий датасет Olivetti / AT&T «Database of Faces» (40 анонімних осіб,
по 10 фотографій 64×64), який завантажує scikit-learn. Групове фото й відео складаються
з фотографій, що НЕ входять до бази розпізнавання (окремі знімки тих самих осіб),
а також з облич двох осіб, яких у базі немає взагалі.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

import cv2
import numpy as np
from numpy.typing import NDArray

from devops_lab7.face_detectors import FaceBox, FaceDetector

Image = NDArray[np.uint8]

NPZ_PATH: Final = Path("data/olivetti_faces.npz")
DB_DIR: Final = Path("data/faces_db")
PHOTO_PATH: Final = Path("data/group_photo.png")
TRUTH_PATH: Final = Path("data/group_photo_truth.json")
VIDEO_PATH: Final = Path("data/video_faces.avi")

KNOWN_SUBJECTS: Final = (1, 7, 9, 13, 20, 26, 31, 34)  # особи в базі розпізнавання
UNKNOWN_SUBJECTS: Final = (11, 38)  # особи, яких у базі немає
DB_IMAGES: Final = tuple(range(8))  # знімки 0–7 кожної особи — база
TEST_IMAGES: Final = (8, 9)  # знімки 8–9 — лише для перевірки (у базу не входять)
PHOTO_IMAGE: Final = 9  # знімок для групового фото

# Розміщення на груповому фото 1280×720: (особа, x центру, y центру, масштаб)
PHOTO_LAYOUT: Final = (
    (13, 170, 205, 2.5),
    (7, 420, 190, 2.5),
    (38, 660, 200, 2.5),
    (26, 890, 190, 2.5),
    (1, 1120, 205, 2.5),
    (20, 290, 470, 2.9),
    (9, 545, 485, 2.9),
    (11, 790, 470, 2.9),
    (34, 1030, 485, 2.9),
    (31, 1185, 500, 2.4),
)


def person_label(subject: int) -> str:
    """Ідентифікатор особи в стилі датасету AT&T: s01 … s40."""
    return f"s{subject + 1:02d}"


def load_olivetti(path: Path = NPZ_PATH) -> tuple[NDArray[np.uint8], NDArray[np.int16]]:
    data = np.load(path)
    return np.asarray(data["images"], dtype=np.uint8), np.asarray(data["target"], dtype=np.int16)


def subject_image(
    images: NDArray[np.uint8], target: NDArray[np.int16], subject: int, index: int
) -> Image:
    return np.asarray(images[np.flatnonzero(target == subject)[index]], dtype=np.uint8)


def background(width: int, height: int, seed: int) -> Image:
    """Тло «стіна кімнати»: вертикальний градієнт + легкий шум (BGR)."""
    rng = np.random.default_rng(seed)
    top, bottom = np.array([214, 222, 228]), np.array([150, 165, 180])
    ramp = np.linspace(0, 1, height)[:, None, None]
    canvas = top * (1 - ramp) + bottom * ramp
    canvas = np.broadcast_to(canvas, (height, width, 3)).copy()
    canvas += rng.normal(0, 4, canvas.shape)
    return np.clip(canvas, 0, 255).astype(np.uint8)


def paste_face(canvas: Image, face: Image, cx: int, cy: int, scale: float) -> FaceBox:
    """Вставляє обличчя (з «плечима») з м'якими краями; повертає рамку вставленого обличчя."""
    size = int(face.shape[0] * scale)
    big = cv2.resize(face, (size, size), interpolation=cv2.INTER_CUBIC)
    x0, y0 = cx - size // 2, cy - size // 2
    # шия та плечі (одяг) під обличчям
    skin = int(face[-8:, 24:40].mean())
    cv2.rectangle(
        canvas,
        (cx - size // 6, y0 + int(size * 0.8)),
        (cx + size // 6, y0 + int(size * 1.12)),
        (skin, skin, skin),
        -1,
    )
    cv2.ellipse(
        canvas,
        (cx, y0 + int(size * 1.5)),
        (int(size * 0.72), int(size * 0.42)),
        0,
        180,
        360,
        (78, 66, 58),
        -1,
        cv2.LINE_AA,
    )
    mask = np.zeros((size, size), np.float32)
    cv2.ellipse(
        mask, (size // 2, size // 2), (int(size * 0.5), int(size * 0.58)), 0, 0, 360, 1.0, -1
    )
    soft = np.asarray(cv2.GaussianBlur(mask, (0, 0), size * 0.03), dtype=np.float32)[..., None]
    h, w = canvas.shape[:2]
    xa, ya, xb, yb = max(0, x0), max(0, y0), min(w, x0 + size), min(h, y0 + size)
    if xa >= xb or ya >= yb:
        return FaceBox(x0, y0, 0, 0)
    region = canvas[ya:yb, xa:xb].astype(np.float32)
    patch = cv2.cvtColor(big, cv2.COLOR_GRAY2BGR)[ya - y0 : yb - y0, xa - x0 : xb - x0]
    alpha = soft[ya - y0 : yb - y0, xa - x0 : xb - x0]
    canvas[ya:yb, xa:xb] = (region * (1 - alpha) + patch * alpha).astype(np.uint8)
    return FaceBox(x0, y0, size, size)


@dataclass(frozen=True)
class TruthFace:
    subject: int
    label: str
    known: bool
    box: tuple[int, int, int, int]


def compose_group_photo(
    images: NDArray[np.uint8], target: NDArray[np.int16]
) -> tuple[Image, list[TruthFace]]:
    """Групове фото 1280×720 з 10 осіб: 8 є в базі (інші знімки), 2 — невідомі."""
    canvas = background(1280, 720, seed=7)
    truth = []
    for subject, cx, cy, scale in PHOTO_LAYOUT:
        index = PHOTO_IMAGE if subject in KNOWN_SUBJECTS else 0
        box = paste_face(canvas, subject_image(images, target, subject, index), cx, cy, scale)
        truth.append(
            TruthFace(
                subject,
                person_label(subject),
                subject in KNOWN_SUBJECTS,
                (box.x, box.y, box.w, box.h),
            )
        )
    return canvas, truth


def build_database(
    images: NDArray[np.uint8], target: NDArray[np.int16], db_dir: Path = DB_DIR
) -> int:
    """Зберігає підготовлені зображення бази: data/faces_db/<особа>/<n>.png."""
    count = 0
    for subject in KNOWN_SUBJECTS:
        folder = db_dir / person_label(subject)
        folder.mkdir(parents=True, exist_ok=True)
        for index in DB_IMAGES:
            cv2.imwrite(str(folder / f"{index}.png"), subject_image(images, target, subject, index))
            count += 1
    return count


def extract_face(raw: Image, detector: FaceDetector) -> Image:
    """Виділяє обличчя з 64×64-знімка так само, як на фото: вставка на тло + детектор.

    Пробуємо кілька масштабів; якщо детектор не спрацював — беремо рамку, розширену
    на 22 % (у середньому така рамка Хаара навколо вставленого обличчя).
    """
    canvas = background(360, 400, seed=1)
    for scale in (2.6, 2.4, 2.8, 3.0, 2.2):
        frame = canvas.copy()
        pasted = paste_face(frame, raw, 180, 170, scale)
        boxes = detector.detect(frame)
        if boxes:
            box = max(boxes, key=lambda b: b.area)
            return np.asarray(cv2.cvtColor(box.crop(frame), cv2.COLOR_BGR2GRAY), dtype=np.uint8)
    grow = int(pasted.w * 0.11)
    box = FaceBox(pasted.x - grow, pasted.y - grow // 2, pasted.w + 2 * grow, pasted.h + 2 * grow)
    return np.asarray(cv2.cvtColor(box.crop(frame), cv2.COLOR_BGR2GRAY), dtype=np.uint8)


def load_database(
    detector: FaceDetector, db_dir: Path = DB_DIR
) -> tuple[list[Image], list[int], dict[int, str]]:
    """Читає базу з диска: обличчя, числові мітки та назви осіб."""
    faces, labels, names = [], [], {}
    for label, folder in enumerate(sorted(p for p in db_dir.iterdir() if p.is_dir())):
        names[label] = folder.name
        for file in sorted(folder.glob("*.png")):
            raw = np.asarray(cv2.imread(str(file), cv2.IMREAD_GRAYSCALE), dtype=np.uint8)
            faces.append(extract_face(raw, detector))
            labels.append(label)
    return faces, labels, names


def save_truth(truth: list[TruthFace], path: Path = TRUTH_PATH) -> None:
    path.write_text(
        json.dumps([asdict(t) for t in truth], ensure_ascii=False, indent=1), encoding="utf-8"
    )


def load_truth(path: Path = TRUTH_PATH) -> list[TruthFace]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [TruthFace(r["subject"], r["label"], r["known"], tuple(r["box"])) for r in raw]


def render_video(
    images: NDArray[np.uint8],
    target: NDArray[np.int16],
    path: Path = VIDEO_PATH,
    fps: int = 20,
    seconds: int = 8,
) -> int:
    """Синтетичний відеофайл 960×540: три особи рухаються, наближаються, з'являються."""
    width, height, frames = 960, 540, fps * seconds
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"MJPG"), fps, (width, height))
    base = background(width, height, seed=11)
    walker = subject_image(images, target, 7, 9)
    zoomer = subject_image(images, target, 20, 8)
    stranger = subject_image(images, target, 38, 3)
    for n in range(frames):
        frame = base.copy()
        t = n / (frames - 1)
        paste_face(frame, walker, int(-80 + t * 1100), 170, 2.0)  # іде зліва направо
        if n >= fps:  # з 1-ї секунди наближається до камери
            k = (n - fps) / (frames - fps)
            paste_face(frame, zoomer, 230, 385, 1.6 + 1.4 * k)
        if 3 * fps <= n < 7 * fps:  # невідома особа в кадрі з 3-ї по 7-му секунду
            paste_face(frame, stranger, 740, 370, 2.5)
        writer.write(frame)
    writer.release()
    return frames
