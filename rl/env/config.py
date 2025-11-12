from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence, Tuple

from rl.env.constants import ACTIONS_PER_COLOR


@dataclass(frozen=True)
class BoardShape:
    """Static geometry of a FlowFree board."""

    width: int
    height: int
    color_count: int

    @property
    def cell_count(self) -> int:
        return self.width * self.height

    @property
    def action_dim(self) -> int:
        return self.color_count * ACTIONS_PER_COLOR


@dataclass(frozen=True)
class RewardPreset:
    """
    Declarative description of reward engines to activate.

    `components` lists engine identifiers (e.g. ("potential", "completion")),
    while `params` captures scalar knobs for the engines.
    """

    name: str
    components: Tuple[str, ...]
    params: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class ObservationSpec:
    """
    Description of the observation tensor emitted by the env package.

    `channels` follow the channel-first convention; auxiliary scalars are
    documented separately in the env implementation.
    """

    channels: Tuple[str, ...]
    dtype: str = "float32"
    include_temporal_planes: bool = True

    def index_of(self, channel: str) -> int:
        return self.channels.index(channel)


@dataclass(frozen=True)
class MaskConfig:
    """Configuration for action mask updates."""

    strategy: str = "incremental"  # values: {"incremental", "full"}
    track_neighbors: bool = True


DEFAULT_STEP_BUFFER = 12


def default_max_steps(width: int, height: int) -> int:
    """Compute the default max-step budget for a board."""
    return width * height + DEFAULT_STEP_BUFFER


@dataclass(frozen=True)
class EnvConfig:
    """
    Immutable configuration for FlowFreeEnv construction.

    - `shape` defines geometry and action space size.
    - `puzzle` is the serialized board with endpoints (x / digit encoding).
    - `max_steps` bounds episode length; defaults to 4 * cell_count.
    - `reward` selects the reward composition; see `rl.env.rewards`.
    - `observation` declares which channels are materialised each step.
    - `mask` controls how the action mask is recomputed.
    - `seed` seeds NumPy/Python RNGs for deterministic resets.
    """

    shape: BoardShape
    puzzle: str
    reward: RewardPreset
    observation: ObservationSpec
    mask: MaskConfig = field(default_factory=MaskConfig)
    max_steps: int | None = None
    seed: int | None = None

    @property
    def effective_max_steps(self) -> int:
        if self.max_steps is not None:
            return self.max_steps
        return default_max_steps(self.shape.width, self.shape.height)


# Default presets ----------------------------------------------------------------

BASE_CHANNELS: Tuple[str, ...] = (
    "occupancy",
    "endpoints",
    "heads",
    "free",
    "congestion",
    "distance",
    "connectivity",
)

DEFAULT_OBSERVATION = ObservationSpec(channels=BASE_CHANNELS)

POTENTIAL_REWARD = RewardPreset(
    name="potential",
    components=("potential", "completion", "constraints"),
    params={
        "move_penalty": -0.05,
        "distance_scale": 0.35,
        "complete_bonus": 1.0,
        "complete_revert_penalty": 3.0,
        "complete_sustain_bonus": 0.05,
        "solve_bonus": 25.0,
        "invalid_penalty": -1.0,
        "dead_pocket_penalty": 0.0,
        "disconnect_penalty": -0.5,
        "degree_penalty": -0.3,
        "unsolved_penalty": -5.0,
        "undo_penalty": -0.1,
    },
)
