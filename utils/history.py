import csv
from pathlib import Path

_METRIC_KEYS = ("loss", "auc_pr", "f1", "iou", "soft_iou", "prec", "rec")


def save_history_csv(history: dict[str, list], path: Path | str) -> None:
    """Overwrite CSV with the full history dict. Call after each epoch for live tracking."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    values = list(history.values())
    n_epochs = len(values[0]) if values else 0
    if n_epochs == 0:
        return

    fieldnames = ["epoch", *history.keys()]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(n_epochs):
            row: dict = {"epoch": i + 1}
            for key, vals in history.items():
                row[key] = round(vals[i], 8)
            writer.writerow(row)


def load_history_csv(path: Path | str) -> dict[str, list[float]]:
    """Load a history CSV back into the same dict format as the trainer produces."""
    path = Path(path)
    history: dict[str, list[float]] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            for key, val in row.items():
                if key == "epoch":
                    continue
                history.setdefault(key, []).append(float(val))
    return history


def plot_history(
    history: dict[str, list],
    out_dir: Path | str,
    prefix: str,
) -> None:
    """Save dashboard PNG + optional loss-parts PNG from history."""
    from utils.viz import plot_dashboard, plot_loss_parts

    out_dir = Path(out_dir)
    plot_dashboard(history, out_dir / f"{prefix}_dashboard.png")
    if any(k.startswith("loss_") and k != "loss" for k in history):
        plot_loss_parts(history, out_dir / f"{prefix}_loss_parts.png")
