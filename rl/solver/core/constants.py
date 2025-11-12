from __future__ import annotations

MAX_COLORS = 12
MAX_SIZE = 10
ACTIONS_PER_COLOR = 5

# Base observation tensor packs occupancy/endpoints/heads plus free & congestion masks & distance & connectivity maps & temporal.
BASE_OBS_CHANNELS = 4 * MAX_COLORS + 2 + 2 * MAX_COLORS + 2  # 76 when MAX_COLORS == 12

# Extra context channels appended during preprocessing (see observation.py).
# - 2 spatial coordinate maps (row/col)
# - 5 global scalars broadcast spatially (step fraction, board fraction, colour fraction,
#   empty-cell fraction, mean remaining fraction)
# - 12 per-colour remaining-length maps (one per potential colour slot)
EXTRA_OBS_CHANNELS = 2 + 5 + MAX_COLORS

MAX_CHANNELS = BASE_OBS_CHANNELS + EXTRA_OBS_CHANNELS
ACTION_DIM = MAX_COLORS * ACTIONS_PER_COLOR

__all__ = [
    "MAX_COLORS",
    "MAX_SIZE",
    "ACTIONS_PER_COLOR",
    "BASE_OBS_CHANNELS",
    "EXTRA_OBS_CHANNELS",
    "MAX_CHANNELS",
    "ACTION_DIM",
]
