"""Datums-invariante Phänologie-Features für eine beliebige Pixelmenge.

Ein Codepfad für Training und Inferenz: `pixel_features` bekommt Zeilen- und
Spaltenindizes und liefert eine Matrix fester Breite. Kein Spaltenname trägt
ein Datum, deshalb hängt weder die Feature-Breite noch das trainierte Modell
an der Auswahl der Aufnahmen.
"""

import calendar
import datetime as dt
import logging
import warnings
from collections.abc import Sequence
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

from deadwood_spectral.align import parse_date
from deadwood_spectral.grid import ReferenceGrid, assert_matches_grid
from deadwood_spectral.indices import BAND_NAMES, compute_indices

logger = logging.getLogger(__name__)

# NIR steht neben den Indizes, weil ein kollabiertes NIR die direkteste
# Totholz-Signatur ist; brightness und green_red kommen aus dem RGB-Komposit.
MEASURES: tuple[str, ...] = ("ndvi", "ndre", "NIR", "brightness", "green_red")
STATS: tuple[str, ...] = ("max", "min", "amplitude", "mean", "std", "greenup_slope")
STATIC: tuple[str, ...] = ("ndsm",)
FEATURE_NAMES: tuple[str, ...] = (
    tuple(f"{measure}_{stat}" for measure in MEASURES for stat in STATS) + STATIC
)


def _shift_months(date: dt.date, months: int) -> dt.date:
    """`date` um `months` Monate zurück, ohne einen 31. Februar zu erfinden."""
    index = date.year * 12 + (date.month - 1) - months
    year, month0 = divmod(index, 12)
    day = min(date.day, calendar.monthrange(year, month0 + 1)[1])
    return dt.date(year, month0 + 1, day)


def window_dates(stack_dir: str | Path, label_date: str, window_months: int) -> list[str]:
    """Aufnahmedaten im Fenster (label_date - window_months, label_date].

    Halboffen: das Ankerdatum gehört dazu, der Fensteranfang nicht. Die Labels
    stammen vom Ankerdatum, deshalb dürfen ältere Zyklen nicht mitgemittelt
    werden.
    """
    anchor = dt.datetime.strptime(label_date, "%Y%m%d").date()
    start = _shift_months(anchor, int(window_months))
    dates = sorted(
        date
        for date in (parse_date(p) for p in Path(stack_dir).glob("*_stack.tif"))
        if start < dt.datetime.strptime(date, "%Y%m%d").date() <= anchor
    )
    if not dates:
        raise ValueError(f"no aligned stack in {stack_dir} inside ({start:%Y%m%d}, {label_date}]")
    logger.info("window %s..%s: %d date(s)", f"{start:%Y%m%d}", label_date, len(dates))
    return dates


def stack_paths(stack_dir: str | Path, dates: Sequence[str]) -> list[Path]:
    """Stack-Pfade zu den Daten; prüft die Existenz *vor* dem ersten Lesen.

    Ein echter Lauf liest ein Dutzend ~800-MB-Stacks; ein erst danach
    auffallender fehlender Pfad wirft die Arbeit weg.
    """
    paths = [Path(stack_dir) / f"{date}_stack.tif" for date in dates]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("missing aligned stack(s): " + ", ".join(missing))
    return paths


def _slope(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Kleinste-Quadrate-Steigung je Zeile über die gültigen Termine.

    Termine gelten als gleich beabstandete Schritte, nicht als Kalendertage:
    die Befliegungen sind grob zweiwöchentlich, und eine Tagesskalierung würde
    von den gelegentlichen langen Lücken dominiert statt von der Phänologie.
    """
    n_dates = values.shape[1]
    x = np.arange(n_dates, dtype=np.float64)
    weights = valid.astype(np.float64)
    n_valid = weights.sum(axis=1)
    safe = np.where(n_valid > 0, n_valid, 1.0)

    y = np.where(valid, np.nan_to_num(values.astype(np.float64), nan=0.0), 0.0)
    x_mean = (weights * x).sum(axis=1) / safe
    y_mean = y.sum(axis=1) / safe

    dx = (x[None, :] - x_mean[:, None]) * weights
    dy = (y - y_mean[:, None]) * weights
    denominator = (dx * dx).sum(axis=1)
    out = np.full(values.shape[0], np.nan, dtype=np.float64)
    np.divide((dx * dy).sum(axis=1), denominator, out=out, where=denominator > 0)
    return out


def aggregate_series(series: dict[str, np.ndarray], min_valid_dates: int = 4) -> pd.DataFrame:
    """Zeitreihen je Messgröße -> 30 datums-invariante Statistiken.

    Gültigkeit wird über die Messgrößen hinweg geteilt: ein Termin, an dem
    auch nur eine Messgröße NaN ist, zählt für keine. Andernfalls würden
    `mean` und `slope` verschiedener Messgrößen über verschiedene Terminmengen
    laufen und wären nicht mehr vergleichbar.
    """
    missing = [m for m in MEASURES if m not in series]
    if missing:
        raise ValueError(f"missing measure(s) in series: {missing}")

    stacked = np.stack([np.asarray(series[m], dtype=np.float64) for m in MEASURES])
    valid = np.asarray(np.isfinite(stacked).all(axis=0))
    n_valid = valid.sum(axis=1)
    enough = n_valid >= max(1, int(min_valid_dates))

    out = pd.DataFrame(index=pd.RangeIndex(stacked.shape[1]))
    for measure, values in zip(MEASURES, stacked):
        masked = np.where(valid, values, np.nan)
        with warnings.catch_warnings():
            # Eine Zeile ohne einen einzigen gültigen Termin ist erwartet, nicht
            # aussergewöhnlich; sie wird unten ohnehin auf NaN gesetzt.
            warnings.simplefilter("ignore", RuntimeWarning)
            v_max = np.nanmax(masked, axis=1)
            v_min = np.nanmin(masked, axis=1)
            v_mean = np.nanmean(masked, axis=1)
            v_std = np.nanstd(masked, axis=1)
        stats = {
            "max": v_max,
            "min": v_min,
            "amplitude": v_max - v_min,
            "mean": v_mean,
            "std": v_std,
            "greenup_slope": _slope(values, valid),
        }
        for stat in STATS:
            out[f"{measure}_{stat}"] = np.where(enough, stats[stat], np.nan)

    dropped = int((~enough).sum())
    if dropped:
        logger.info(
            "%d/%d pixel(s) with fewer than %d valid date(s) -> NaN features",
            dropped,
            len(enough),
            min_valid_dates,
        )
    return out


def measure_series(
    stack_paths_: Sequence[Path],
    grid: ReferenceGrid,
    rows: np.ndarray,
    cols: np.ndarray,
    chunk_rows: int = 512,
) -> dict[str, np.ndarray]:
    """Rohe Zeitreihe je Messgroesse fuer eine Pixelmenge -> {measure: (n_px, n_dates)}.

    Gelesen wird in Zeilen-Chunks ueber alle Aufnahmen gleichzeitig, damit die
    Zeitreihe eines Pixels entsteht, ohne je einen ganzen Stack im Speicher zu
    halten. Getrennt von `pixel_features`, weil der Phaenologie-Plot den rohen
    Verlauf braucht und nicht die Aggregate.
    """
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    if rows.shape != cols.shape:
        raise ValueError(f"rows/cols length mismatch: {rows.shape} vs {cols.shape}")
    n_dates = len(stack_paths_)
    if n_dates == 0:
        raise ValueError("no stacks given")

    series = {m: np.full((rows.size, n_dates), np.nan, dtype=np.float32) for m in MEASURES}
    order = np.argsort(rows, kind="stable")

    with ExitStack() as stack:
        sources = []
        for path in stack_paths_:
            src = stack.enter_context(rasterio.open(path))
            assert_matches_grid(src, grid, str(path))
            sources.append(src)

        for start in range(0, grid.height, chunk_rows):
            stop = min(start + chunk_rows, grid.height)
            sel = order[(rows[order] >= start) & (rows[order] < stop)]
            if sel.size == 0:
                continue
            local_rows, local_cols = rows[sel] - start, cols[sel]
            window = Window.from_slices((start, stop), (0, grid.width))
            for date_idx, src in enumerate(sources):
                block = src.read(window=window).astype(np.float32)
                names = [d or n for d, n in zip(src.descriptions, BAND_NAMES)]
                values = dict(zip(names, block))
                values.update(compute_indices(block, names))
                for measure in MEASURES:
                    series[measure][sel, date_idx] = values[measure][local_rows, local_cols]
    return series


def pixel_features(
    stack_paths_: Sequence[Path],
    grid: ReferenceGrid,
    rows: np.ndarray,
    cols: np.ndarray,
    ndsm_path: str | Path | None = None,
    chunk_rows: int = 512,
    min_valid_dates: int = 4,
) -> pd.DataFrame:
    """Feature-Matrix für eine beliebige Pixelmenge — ein Pfad für Training und Inferenz.

    Gelesen wird in Zeilen-Chunks über alle Aufnahmen gleichzeitig, damit die
    Zeitreihe eines Pixels entsteht, ohne je einen ganzen Stack im Speicher zu
    halten. Die Ausgabe steht in Eingabereihenfolge, nicht in Chunk-Reihenfolge.
    """
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    series = measure_series(stack_paths_, grid, rows, cols, chunk_rows=chunk_rows)
    n_pixels = rows.size
    out = aggregate_series(series, min_valid_dates=min_valid_dates)

    if ndsm_path is None:
        out["ndsm"] = np.full(n_pixels, np.nan, dtype=np.float64)
    else:
        heights = np.full(n_pixels, np.nan, dtype=np.float64)
        order = np.argsort(rows, kind="stable")
        with rasterio.open(ndsm_path) as src:
            assert_matches_grid(src, grid, str(ndsm_path))
            for start in range(0, grid.height, chunk_rows):
                stop = min(start + chunk_rows, grid.height)
                sel = order[(rows[order] >= start) & (rows[order] < stop)]
                if sel.size == 0:
                    continue
                block = src.read(1, window=Window.from_slices((start, stop), (0, grid.width)))
                heights[sel] = block[rows[sel] - start, cols[sel]]
        out["ndsm"] = heights

    return out[list(FEATURE_NAMES)]
