"""Named input-channel resolution.

Single owner of all channel-name logic: the ordered spectral names of a
patch/scene stack (channels.json manifest) plus the train config's
``dataset.input_channels`` resolve to raster band indexes, normalisation-stat
subsets, and the mapping of pretrained R/G/B encoder weights onto input
positions. Dataset, model builder, trainer, and predict all consume one
ChannelSpec instead of doing their own index math.
"""

import json
from pathlib import Path

import numpy as np

# First-conv column order of RGB-pretrained encoders
PRETRAINED_SLOTS = ("red", "green", "blue")
# Reserved name for the DSM-derived input channel (own patch file, not in the stack)
NDSM = "ndsm"


def _require_name_list(value, context: str) -> list[str]:
    """Reject a bare string, which would otherwise iterate character by character.

    A YAML list written without its brackets (``input_channels: red, green``)
    arrives here as a string; splitting it into letters yields a baffling error
    far from the actual typo.
    """
    if isinstance(value, str):
        raise ValueError(f"{context} must be a list of channel names, got the string {value!r}")
    return [str(n) for n in value]


def _validate_unique_names(names: list, context: str) -> list[str]:
    """Validate that a list of channel names contains no duplicates.

    Args:
        names: list of channel names to validate
        context: context string for error message (e.g., "stack_names" or path)

    Returns:
        List of stringified names if validation passes.

    Raises:
        ValueError: if names is not a list, is empty, or contains duplicates.
    """
    names_str = _require_name_list(names, context)
    if not names_str or len(set(names_str)) != len(names_str):
        raise ValueError(f"{context} must list unique channel names, got {names_str}")
    return names_str


def load_manifest(path: Path | str) -> list[str]:
    """Read the ordered channel names from a channels.json manifest."""
    names = json.loads(Path(path).read_text())["names"]
    return _validate_unique_names(names, str(path))


class ChannelSpec:
    """Resolves named model input channels against a stack manifest.

    stack_names: spectral channel names of the stack image, in band order.
    input_channels: model input channels in order; may include NDSM anywhere.
    pretrained_channel_map: optional {slot: channel_name | None}; overrides the
        name convention per slot (None suppresses that slot's assignment).
    """

    def __init__(
        self,
        stack_names: list[str],
        input_channels: list[str],
        pretrained_channel_map: dict | None = None,
    ):
        self.stack_names = _validate_unique_names(stack_names, "stack_names")
        self.input_channels = _require_name_list(input_channels, "input_channels")

        if NDSM in self.stack_names:
            raise ValueError(f"'{NDSM}' is reserved and cannot be a stack channel")
        if len(set(self.input_channels)) != len(self.input_channels):
            raise ValueError(f"input_channels contains duplicates: {self.input_channels}")
        available = set(self.stack_names) | {NDSM}
        unknown = [c for c in self.input_channels if c not in available]
        if unknown:
            raise ValueError(
                f"Unknown input_channels {unknown!r}; available: {sorted(available)}"
            )

        self._map = dict(pretrained_channel_map) if pretrained_channel_map else {}
        bad_slots = [s for s in self._map if s not in PRETRAINED_SLOTS]
        if bad_slots:
            raise ValueError(
                f"pretrained_channel_map keys must be in {PRETRAINED_SLOTS}, got {bad_slots}"
            )
        bad_names = [
            v for v in self._map.values() if v is not None and v not in self.input_channels
        ]
        if bad_names:
            raise ValueError(
                f"pretrained_channel_map values {bad_names} not in input_channels"
            )

        self.pretrained_assignment = self._resolve_pretrained()

    # ------------------------------------------------------------- selection

    @property
    def in_channels(self) -> int:
        return len(self.input_channels)

    @property
    def use_ndsm(self) -> bool:
        return NDSM in self.input_channels

    @property
    def ndsm_position(self) -> int | None:
        """Input position of the ndsm channel, or None if it is not selected."""
        return self.input_channels.index(NDSM) if self.use_ndsm else None

    @property
    def display_rgb_positions(self) -> list[int]:
        """Three input positions to render as R, G, B in a pseudo-RGB preview.

        Prefers the channels that won the pretrained R/G/B slots — true colour
        when red/green/blue are selected, the *_ms equivalents otherwise. Falls
        back to the first spectral channels (last one repeated) when fewer than
        three slots are filled, so previews work for any channel selection.
        """
        assignment = self.pretrained_assignment
        if len(assignment) == len(PRETRAINED_SLOTS):
            by_slot = {slot: pos for pos, slot in assignment.items()}
            return [by_slot[s] for s in range(len(PRETRAINED_SLOTS))]
        spectral = [i for i, c in enumerate(self.input_channels) if c != NDSM]
        if not spectral:
            raise ValueError("no spectral channel to preview; input_channels is ndsm-only")
        pos = spectral[:3]
        return pos + [pos[-1]] * (3 - len(pos))

    @property
    def stack_indexes(self) -> list[int]:
        """1-based rasterio band indexes of the selected spectral channels, in input order."""
        return [self.stack_names.index(c) + 1 for c in self.input_channels if c != NDSM]

    # ------------------------------------------------------------ pretrained

    def _resolve_pretrained(self) -> dict[int, int]:
        """{input position: slot index} — exact name wins, else unique '<slot>_*' prefix."""
        assignment: dict[int, int] = {}
        for slot_idx, slot in enumerate(PRETRAINED_SLOTS):
            if slot in self._map:
                name = self._map[slot]
                if name is None:
                    continue
            elif slot in self.input_channels:
                name = slot
            else:
                prefixed = [
                    c for c in self.input_channels
                    if c != NDSM and c.startswith(slot + "_")
                ]
                if len(prefixed) > 1:
                    raise ValueError(
                        f"Channels {prefixed} both match pretrained slot '{slot}'; "
                        "set model.pretrained_channel_map to disambiguate"
                    )
                if not prefixed:
                    continue
                name = prefixed[0]
            assignment[self.input_channels.index(name)] = slot_idx
        return assignment

    # -------------------------------------------------------------- assembly

    def assemble(self, stack: np.ndarray, ndsm: np.ndarray | None) -> np.ndarray:
        """Combine stack bands (read with indexes=stack_indexes) and the ndsm
        array into a (in_channels, H, W) array in input_channels order."""
        if not self.use_ndsm:
            return stack
        if ndsm is None:
            raise ValueError("input_channels selects ndsm but no ndsm array was given")
        pos = self.input_channels.index(NDSM)
        return np.insert(stack, pos, ndsm[0], axis=0)

    # ----------------------------------------------------------------- stats

    def norm_stats(self, stats: dict) -> dict[str, list[float]]:
        """Subset full stats {names, mean, std} to input_channels, in input order."""
        if "names" not in stats:
            raise ValueError(
                "train_stats.json has no 'names' field — regenerate it with "
                "scripts/preprocess.py (old-format stats are not supported)"
            )
        by_name = {n: i for i, n in enumerate(stats["names"])}
        missing = [c for c in self.input_channels if c not in by_name]
        if missing:
            raise ValueError(f"stats lack channels {missing}; available: {stats['names']}")
        idx = [by_name[c] for c in self.input_channels]
        return {
            "mean": [float(stats["mean"][i]) for i in idx],
            "std": [float(stats["std"][i]) for i in idx],
        }

    # -------------------------------------------------------------- freezing

    def frozen_indices(self, names: list[str] | None) -> list[int]:
        """Input positions for frozen_pretrained_channels names, validated."""
        names = list(names or [])
        unknown = [n for n in names if n not in self.input_channels]
        if unknown:
            raise ValueError(
                f"frozen_pretrained_channels {unknown} not in input_channels "
                f"{self.input_channels}"
            )
        idx = [self.input_channels.index(n) for n in names]
        unpretrained = [n for n, i in zip(names, idx) if i not in self.pretrained_assignment]
        if unpretrained:
            raise ValueError(
                f"frozen_pretrained_channels {unpretrained} did not receive "
                "pretrained weights — freezing them would freeze random init"
            )
        return idx
