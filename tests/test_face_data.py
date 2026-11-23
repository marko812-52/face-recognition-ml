"""Тести підготовлених даних для завдань 7–9."""

from pathlib import Path

from devops_lab7.face_data import (
    DB_IMAGES,
    KNOWN_SUBJECTS,
    build_database,
    compose_group_photo,
    load_olivetti,
    person_label,
)


def test_olivetti_shape() -> None:
    images, target = load_olivetti()
    assert images.shape == (400, 64, 64)
    assert len(set(target.tolist())) == 40


def test_group_photo_has_known_and_unknown_people() -> None:
    images, target = load_olivetti()
    photo, truth = compose_group_photo(images, target)
    assert photo.shape == (720, 1280, 3)
    assert len(truth) == 10
    assert sum(t.known for t in truth) == len(KNOWN_SUBJECTS)
    assert all(t.box[0] >= 0 and t.box[0] + t.box[2] <= 1280 for t in truth)


def test_build_database(tmp_path: Path) -> None:
    images, target = load_olivetti()
    count = build_database(images, target, tmp_path)
    assert count == len(KNOWN_SUBJECTS) * len(DB_IMAGES)
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(
        person_label(s) for s in KNOWN_SUBJECTS
    )


def test_person_label() -> None:
    assert person_label(0) == "s01"
    assert person_label(39) == "s40"
