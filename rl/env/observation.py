from __future__ import annotations

from typing import Callable, Dict, Iterable, List

import numpy as np

from rl.env.config import ObservationSpec
from rl.env.constants import ACTIONS_PER_COLOR, EMPTY
from rl.env.state import BoardState, manhattan

ChannelFactory = Callable[[BoardState], np.ndarray]


def _occupancy_planes(state: BoardState) -> np.ndarray:
    planes = np.zeros((state.shape.color_count, state.shape.height, state.shape.width), dtype=np.float32)
    for idx, value in enumerate(state.cells):
        if value == EMPTY:
            continue
        color = value - 1
        r, c = divmod(idx, state.shape.width)
        planes[color, r, c] = 1.0
    return planes


def _endpoints_planes(state: BoardState) -> np.ndarray:
    planes = np.zeros((state.shape.color_count, state.shape.height, state.shape.width), dtype=np.float32)
    for color in range(state.shape.color_count):
        start = state.start_positions[color]
        target = state.target_positions[color]
        sr, sc = divmod(start, state.shape.width)
        tr, tc = divmod(target, state.shape.width)
        planes[color, sr, sc] = 1.0
        planes[color, tr, tc] = 1.0
    return planes


def _heads_planes(state: BoardState) -> np.ndarray:
    planes = np.zeros((state.shape.color_count * 2, state.shape.height, state.shape.width), dtype=np.float32)
    for color in range(state.shape.color_count):
        head = state.head_positions[color]
        target = state.target_positions[color]
        hr, hc = divmod(head, state.shape.width)
        tr, tc = divmod(target, state.shape.width)
        planes[color * 2, hr, hc] = 1.0
        planes[color * 2 + 1, tr, tc] = 1.0
    return planes


def _free_plane(state: BoardState) -> np.ndarray:
    plane = np.zeros((1, state.shape.height, state.shape.width), dtype=np.float32)
    for idx, value in enumerate(state.cells):
        if value == EMPTY:
            r, c = divmod(idx, state.shape.width)
            plane[0, r, c] = 1.0
    return plane


def _congestion_plane(state: BoardState) -> np.ndarray:
    plane = np.zeros((1, state.shape.height, state.shape.width), dtype=np.float32)
    for idx, value in enumerate(state.cells):
        if value != EMPTY:
            continue
        r, c = divmod(idx, state.shape.width)
        count = 0
        for nb in state.neighbors(idx):
            if state.cells[nb] == EMPTY:
                count += 1
        plane[0, r, c] = count
    return plane


def _distance_planes(state: BoardState) -> np.ndarray:
    planes = np.zeros((state.shape.color_count, state.shape.height, state.shape.width), dtype=np.float32)
    max_distance = state.shape.height + state.shape.width
    if max_distance == 0:
        return planes
    for color in range(state.shape.color_count):
        target = state.target_positions[color]
        tr, tc = divmod(target, state.shape.width)
        for r in range(state.shape.height):
            for c in range(state.shape.width):
                planes[color, r, c] = (abs(r - tr) + abs(c - tc)) / max_distance
    return planes


def _connectivity_planes(state: BoardState) -> np.ndarray:
    planes = np.zeros((state.shape.color_count, state.shape.height, state.shape.width), dtype=np.float32)
    for color in range(state.shape.color_count):
        head = state.head_positions[color]
        allowed = {EMPTY, color + 1}
        visited = set([head])
        stack = [head]
        while stack:
            node = stack.pop()
            r, c = divmod(node, state.shape.width)
            planes[color, r, c] = 1.0
            for nb in state.neighbors(node):
                if nb in visited:
                    continue
                if state.cells[nb] in allowed:
                    visited.add(nb)
                    stack.append(nb)
    return planes


def _temporal_planes(state: BoardState, last_action: int | None) -> np.ndarray:
    plane = np.zeros((2, state.shape.height, state.shape.width), dtype=np.float32)
    if last_action is None:
        return plane
    color = last_action // ACTIONS_PER_COLOR
    slot = last_action % ACTIONS_PER_COLOR
    plane[0, :, :] = color / max(1, state.shape.color_count)
    plane[1, :, :] = slot / float(ACTIONS_PER_COLOR)
    return plane


CHANNEL_REGISTRY: Dict[str, Callable[[BoardState], np.ndarray]] = {
    "occupancy": _occupancy_planes,
    "endpoints": _endpoints_planes,
    "heads": _heads_planes,
    "free": _free_plane,
    "congestion": _congestion_plane,
    "distance": _distance_planes,
    "connectivity": _connectivity_planes,
}


class ObservationBuilder:
    def __init__(self, spec: ObservationSpec):
        self.spec = spec

    def build(
        self,
        state: BoardState,
        mask: np.ndarray,
        *,
        last_action: int | None,
        max_steps: int,
    ) -> dict:
        planes: List[np.ndarray] = []
        for channel in self.spec.channels:
            factory = CHANNEL_REGISTRY.get(channel)
            if factory is None:
                raise ValueError(f"Unknown observation channel '{channel}'")
            planes.append(factory(state))

        if self.spec.include_temporal_planes:
            planes.append(_temporal_planes(state, last_action))

        tensor = np.concatenate(planes, axis=0).astype(self.spec.dtype)

        observation = {
            "tensor": tensor,
            "action_mask": mask.copy(),
            "remaining_lengths": np.asarray(state.distance_to_target, dtype=np.int32),
            "steps": state.steps,
            "height": state.shape.height,
            "width": state.shape.width,
            "color_count": state.shape.color_count,
            "max_steps": max_steps,
        }
        return observation


__all__ = ["ObservationBuilder", "CHANNEL_REGISTRY"]
