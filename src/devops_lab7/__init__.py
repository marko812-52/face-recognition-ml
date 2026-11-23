"""ЛР7 «Класичний ML та CV», варіант 2 — датасет Digits (Ворок М. В., КБ-212)."""


def main() -> None:
    """Запускає всі частини роботи по черзі (без вікон)."""
    from devops_lab7 import detect_photo, detect_video, digits_task, lsp_demo, recognize_photo

    digits_task.main([])
    detect_photo.main([])
    detect_video.main([])
    recognize_photo.main([])
    lsp_demo.main()
