"""Завдання 1–6: класифікація датасету Digits методами SVM і Random Forest.

Запуск:
    uv run lab7-digits            # звіти, матриці помилок, час fit/predict, графіки у output/
    uv run lab7-digits --show 8   # додатково показати вікна з графіками на 8 секунд
"""

import argparse
import json
from collections.abc import Sequence

import matplotlib

from devops_lab7.console import OUTPUT_DIR, section, setup_console, table
from devops_lab7.ml_models import (
    ClassifierModel,
    DatasetSplit,
    EvaluationResult,
    RandomForestModel,
    SvmModel,
    evaluate,
    load_digits_split,
)


def build_models(only: str | None = None) -> list[ClassifierModel]:
    """Два обрані методи. Завдяки LSP сюди можна додати будь-який інший підклас."""
    models: dict[str, ClassifierModel] = {"svm": SvmModel(), "rf": RandomForestModel()}
    return [model for key, model in models.items() if only in (None, key)]


def print_confusion(result: EvaluationResult) -> None:
    labels = [str(i) for i in range(result.confusion.shape[0])]
    rows = [[f"{labels[i]} |", *row.tolist()] for i, row in enumerate(result.confusion)]
    print(table(["факт\\прогноз", *labels], rows))


def summary_rows(results: Sequence[EvaluationResult]) -> list[list[str]]:
    rows = []
    for r in results:
        macro = r.report["macro avg"]
        rows.append(
            [
                r.model_name,
                f"{r.accuracy:.4f}",
                f"{macro['precision']:.4f}",
                f"{macro['recall']:.4f}",
                f"{macro['f1-score']:.4f}",
                f"{r.fit_seconds * 1000:.1f}",
                f"{r.predict_seconds * 1000:.2f}",
                str(int(r.confusion.sum() - r.confusion.trace())),
            ]
        )
    return rows


SUMMARY_HEADER = [
    "Модель",
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "fit, мс",
    "predict, мс",
    "Помилок",
]


def save_figures(results: Sequence[EvaluationResult], split: DatasetSplit, show: float) -> None:
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # 1) матриці помилок
    fig, axes = plt.subplots(1, len(results), figsize=(6.2 * len(results), 5.4))
    for ax, r in zip(axes, results, strict=True):
        im = ax.imshow(r.confusion, cmap="Blues")
        for i in range(r.confusion.shape[0]):
            for j in range(r.confusion.shape[1]):
                value = int(r.confusion[i, j])
                color = "white" if value > r.confusion.max() / 2 else "black"
                ax.text(j, i, str(value), ha="center", va="center", color=color, fontsize=9)
        ax.set_title(f"{r.model_name}: accuracy = {r.accuracy:.4f}")
        ax.set_xlabel("Передбачений клас")
        ax.set_ylabel("Справжній клас")
        ax.set_xticks(range(10))
        ax.set_yticks(range(10))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Матриці помилок (Confusion Matrix), датасет Digits, тестова вибірка")
    fig.tight_layout()
    fig.canvas.manager.set_window_title("ЛР7. Матриці помилок")  # type: ignore[union-attr]
    fig.savefig(OUTPUT_DIR / "digits_confusion.png", dpi=130)

    # 2) порівняння метрик і часу
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    metrics = ["precision", "recall", "f1-score"]
    width = 0.8 / len(results)
    for k, r in enumerate(results):
        values = [r.report["macro avg"][m] for m in metrics]
        bars = ax1.bar([i + k * width for i in range(3)], values, width, label=r.model_name)
        ax1.bar_label(bars, fmt="%.3f", fontsize=8)
    centers = [i + width * (len(results) - 1) / 2 for i in range(3)]
    ax1.set_xticks(centers, ["Precision", "Recall", "F1"])
    ax1.set_ylim(0.9, 1.0)
    ax1.set_title("Метрики (macro avg)")
    ax1.legend(loc="lower right")
    ax1.grid(axis="y", alpha=0.3)
    names = [r.model_name for r in results]
    fit_ms = [r.fit_seconds * 1000 for r in results]
    pred_ms = [r.predict_seconds * 1000 for r in results]
    x = range(len(results))
    b1 = ax2.bar([i - 0.2 for i in x], fit_ms, 0.4, label="fit (навчання)")
    b2 = ax2.bar([i + 0.2 for i in x], pred_ms, 0.4, label="predict (передбачення)")
    ax2.bar_label(b1, fmt="%.1f", fontsize=8)
    ax2.bar_label(b2, fmt="%.2f", fontsize=8)
    ax2.set_xticks(list(x), names)
    ax2.set_ylabel("мс (медіана 5 запусків)")
    ax2.set_title("Час навчання і передбачення")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)
    fig2.tight_layout()
    fig2.canvas.manager.set_window_title("ЛР7. Порівняння методів")  # type: ignore[union-attr]
    fig2.savefig(OUTPUT_DIR / "digits_comparison.png", dpi=130)

    # 3) приклади помилкових класифікацій
    fig3, grid = plt.subplots(2, 8, figsize=(12, 3.6))
    for row, r in zip(grid, results, strict=True):
        wrong = [i for i in range(len(split.y_test)) if r.predictions[i] != split.y_test[i]][:8]
        for ax, idx in zip(row, wrong + [None] * (8 - len(wrong)), strict=True):
            ax.axis("off")
            if idx is None:
                continue
            ax.imshow(split.images_test[idx], cmap="gray_r")
            ax.set_title(f"{split.y_test[idx]}→{r.predictions[idx]}", fontsize=10, color="crimson")
        row[0].text(
            -0.5,
            0.5,
            r.model_name,
            transform=row[0].transAxes,
            ha="right",
            va="center",
            fontsize=10,
        )
    fig3.suptitle("Приклади помилок на тестовій вибірці (справжній → передбачений)")
    fig3.tight_layout()
    fig3.canvas.manager.set_window_title("ЛР7. Помилки класифікації")  # type: ignore[union-attr]
    fig3.savefig(OUTPUT_DIR / "digits_errors.png", dpi=130)

    if show:
        plt.show(block=False)
        plt.pause(show)
    plt.close("all")


def main(argv: Sequence[str] | None = None) -> None:
    setup_console()
    parser = argparse.ArgumentParser(prog="lab7-digits", description="ЛР7, завдання 1–6")
    parser.add_argument("--show", type=float, default=0, help="показати вікна N секунд")
    parser.add_argument("--only", choices=["svm", "rf"], help="лише один метод (без графіків)")
    args = parser.parse_args(argv)

    split = load_digits_split()
    section("ЛР7, завдання 1–2. Датасет Digits (варіант 2), поділ на вибірки")
    print(
        f"Зображень: {len(split.y_train) + len(split.y_test)} (8x8 пікселів, 64 ознаки), "
        f"класів: {len(split.target_names)} (цифри 0–9)"
    )
    print(
        f"Навчальна вибірка: {len(split.y_train)}, тестова: {len(split.y_test)} "
        "(25 %, стратифікований поділ, random_state=42)"
    )

    results = []
    for model in build_models(args.only):
        result = evaluate(model, split)
        results.append(result)
        section(f"Завдання 3–5. {model.name}")
        print(f"Модель: {model.describe()}")
        print(
            f"Час навчання (fit): {result.fit_seconds * 1000:.1f} мс; "
            f"час передбачення (predict, {len(split.y_test)} зразків): "
            f"{result.predict_seconds * 1000:.2f} мс"
        )
        print("\nClassification Report:")
        print(result.report_text)
        print("Confusion Matrix (рядки — справжній клас, стовпці — передбачений):")
        print_confusion(result)

    if args.only:
        return
    section("Завдання 6. Порівняння методів (macro avg, час — медіана 5 запусків)")
    print(table(SUMMARY_HEADER, summary_rows(results)))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics = {
        r.model_name: {
            "accuracy": r.accuracy,
            "fit_ms": r.fit_seconds * 1000,
            "predict_ms": r.predict_seconds * 1000,
            "macro_avg": r.report["macro avg"],
            "confusion": r.confusion.tolist(),
        }
        for r in results
    }
    (OUTPUT_DIR / "digits_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    save_figures(results, split, args.show)
    print(
        "\nЗбережено: output/digits_metrics.json, digits_confusion.png, "
        "digits_comparison.png, digits_errors.png"
    )


if __name__ == "__main__":
    main()
