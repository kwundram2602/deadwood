"""Datums-invariante Phänologie-Features für eine beliebige Pixelmenge.

Ein Codepfad für Training und Inferenz: `pixel_features` bekommt Zeilen- und
Spaltenindizes und liefert eine Matrix fester Breite. Kein Spaltenname trägt
ein Datum, deshalb hängt weder die Feature-Breite noch das trainierte Modell
an der Auswahl der Aufnahmen.
"""

import calendar
import datetime as dt
import logging
from collections.abc import Sequence
from pathlib import Path

from deadwood_spectral.align import parse_date

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
