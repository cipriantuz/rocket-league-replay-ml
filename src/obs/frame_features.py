"""Global frame feature extractor: GameState v2 -> fixed-size float32 vector.

Output dimensions depend on player count (1v1/2v2/3v3). Use Mode2v2 (default)
for 2v2 modeling — feature size = 75 (3 ball + 4*18 cars + boost_pad_summary).

The extractor is the SHARED contract between offline training (replay-derived
GameState via rlgym-tools) and live inference (RLBot GameTickPacket via
rlgym-compat). Feature parity is guaranteed by both sources producing the same
GameState dataclass — this module is the single feature definition.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# rlgym is only needed for replay PARSING / feature EXTRACTION (ingest, on Windows).
# The training environment reads already-normalized Parquet and never calls
# the extraction functions below — it only needs this module importable (e.g.
# universal_feature_size). So import rlgym lazily, with fixed RL field-constant
# fallbacks (RLBot Wiki values) when the rlgym/rocketsim stack is absent.
try:
    from rlgym.rocket_league.api import Car, GameState
    from rlgym.rocket_league.common_values import (
        BACK_WALL_Y,
        BLUE_TEAM,
        CEILING_Z,
        ORANGE_TEAM,
        SIDE_WALL_X,
    )
except ImportError:  # training-only env without the rlgym/rocketsim stack
    Car = GameState = object  # placeholders for type hints; extraction unused here
    SIDE_WALL_X = 4096.0
    BACK_WALL_Y = 5120.0
    CEILING_Z = 2044.0
    BLUE_TEAM = 0
    ORANGE_TEAM = 1

# Normalization constants (Unreal units → roughly [-1, 1] range)
POS_NORM = np.array([SIDE_WALL_X, BACK_WALL_Y, CEILING_Z], dtype=np.float32)
VEL_NORM_LINEAR = 2300.0  # MAX_CAR_SPEED ≈ 2300 uu/s
VEL_NORM_ANGULAR = 5.5  # CAR_MAX_ANG_SPEED ≈ 5.5 rad/s
BALL_VEL_NORM = 6000.0  # ball can exceed car speed via boosts/aerials


@dataclass(frozen=True)
class ExtractorConfig:
    n_cars: int = 4  # 2v2 default; 1v1=2, 3v3=6
    include_boost_pads: bool = True
    inverted: bool = False  # team-symmetric view: orange perspective sees flipped field


def physics_features(pos: np.ndarray, lin_vel: np.ndarray, ang_vel: np.ndarray) -> np.ndarray:
    """Pack & normalize a single physics object into a 9-float vector."""
    out = np.empty(9, dtype=np.float32)
    out[0:3] = pos / POS_NORM
    out[3:6] = lin_vel / BALL_VEL_NORM  # both ball and car capped here for simplicity
    out[6:9] = ang_vel / VEL_NORM_ANGULAR
    return out


def ball_features(state: GameState, *, inverted: bool) -> np.ndarray:
    """9 floats: ball position + linear vel + angular vel (normalized, optionally inverted)."""
    ball = state.inverted_ball if inverted else state.ball
    return physics_features(ball.position, ball.linear_velocity, ball.angular_velocity)


def car_features(car: Car, *, inverted: bool) -> np.ndarray:
    """18 floats per car: physics (9) + state (9).

    State: boost_amount, on_ground, has_flip, has_jumped, is_supersonic,
           air_time_since_jump (clipped), flip_time, demo_respawn_timer,
           is_orange (team indicator).
    """
    phys = car._inverted_physics if (inverted and car._inverted_physics is not None) else car.physics
    if inverted and car._inverted_physics is None:
        phys = phys.inverted()
    p = physics_features(phys.position, phys.linear_velocity, phys.angular_velocity)
    s = np.array(
        [
            car.boost_amount / 100.0,
            float(car.on_ground),
            float(car.has_flip),
            float(car.has_jumped),
            float(car.is_supersonic),
            min(car.air_time_since_jump, 1.5) / 1.5,
            min(car.flip_time, 1.0),
            min(car.demo_respawn_timer, 3.0) / 3.0,
            float(car.team_num == ORANGE_TEAM),
        ],
        dtype=np.float32,
    )
    return np.concatenate([p, s])


def boost_pad_summary(state: GameState, *, inverted: bool) -> np.ndarray:
    """Compact boost pad summary: 6 large pad timers + small pad count_ready.

    BallChasing-derived data may have boost_pad_timers shape (34,) where
    indices 3, 4, 15, 18, 29, 30 are large pads (per BOOST_LOCATIONS in
    rlgym-rocket-league common_values). We summarize to 7 floats.
    """
    timers = state.inverted_boost_pad_timers if inverted else state.boost_pad_timers
    if timers is None or len(timers) == 0:
        return np.zeros(7, dtype=np.float32)
    # Indices for the 6 large boost pads in standard field layout.
    large_indices = [3, 4, 15, 18, 29, 30]
    if len(timers) < 34:
        # Fall back gracefully if timer array shape is different.
        large_indices = [i for i in large_indices if i < len(timers)]
    large = np.array([timers[i] for i in large_indices], dtype=np.float32) / 10.0
    if len(large) < 6:
        large = np.pad(large, (0, 6 - len(large)))
    small_mask = np.ones(len(timers), dtype=bool)
    for i in large_indices:
        if i < len(small_mask):
            small_mask[i] = False
    small_ready_count = float((timers[small_mask] <= 1e-3).sum()) / max(small_mask.sum(), 1)
    return np.concatenate([large, np.array([small_ready_count], dtype=np.float32)])


def extract_frame_features(
    state: GameState,
    *,
    config: ExtractorConfig = ExtractorConfig(),
) -> np.ndarray:
    """Build a fixed-length feature vector for a single GameState frame.

    Output size:
        ball (9) + n_cars * car (18) + boost_summary (7 if enabled, else 0)
        = 9 + n_cars*18 + 7  ->  2v2 default = 88

    Cars are emitted in a stable order: blue cars first (sorted by AgentID),
    then orange cars (sorted by AgentID). Missing cars are zero-padded.
    """
    parts = [ball_features(state, inverted=config.inverted)]

    # Sort by AgentID (the stable dict key from rlgym-tools, = player unique_id,
    # constant across all frames of a match). Sorting by id(c) (memory address)
    # was a BUG: it reshuffled which player landed in car_blue0/blue1 every frame,
    # producing physically-impossible teleporting per-car trajectories. str() keeps
    # ordering type-safe across replay (int keys) and live rlgym-compat keys.
    cars_blue = [
        c for _, c in sorted(
            ((aid, c) for aid, c in state.cars.items() if c.team_num == BLUE_TEAM),
            key=lambda kv: str(kv[0]),
        )
    ]
    cars_orange = [
        c for _, c in sorted(
            ((aid, c) for aid, c in state.cars.items() if c.team_num == ORANGE_TEAM),
            key=lambda kv: str(kv[0]),
        )
    ]
    n_per_team = config.n_cars // 2
    all_cars = (cars_blue + [None] * n_per_team)[:n_per_team] + (
        cars_orange + [None] * n_per_team
    )[:n_per_team]

    zero_car = np.zeros(18, dtype=np.float32)
    for car in all_cars:
        parts.append(zero_car if car is None else car_features(car, inverted=config.inverted))

    if config.include_boost_pads:
        parts.append(boost_pad_summary(state, inverted=config.inverted))

    return np.concatenate(parts).astype(np.float32)


def feature_size(config: ExtractorConfig = ExtractorConfig()) -> int:
    """Compute output vector size without invoking the extractor."""
    return 9 + config.n_cars * 18 + (7 if config.include_boost_pads else 0)


def feature_names(config: ExtractorConfig = ExtractorConfig()) -> list[str]:
    """Human-readable names for every feature index — for analysis & viz."""
    names: list[str] = []
    for axis in ("x", "y", "z"):
        names.append(f"ball_pos_{axis}")
    for axis in ("x", "y", "z"):
        names.append(f"ball_linvel_{axis}")
    for axis in ("x", "y", "z"):
        names.append(f"ball_angvel_{axis}")
    for i in range(config.n_cars):
        team = "blue" if i < config.n_cars // 2 else "orange"
        idx_in_team = i if team == "blue" else i - config.n_cars // 2
        prefix = f"car_{team}{idx_in_team}"
        for axis in ("x", "y", "z"):
            names.append(f"{prefix}_pos_{axis}")
        for axis in ("x", "y", "z"):
            names.append(f"{prefix}_linvel_{axis}")
        for axis in ("x", "y", "z"):
            names.append(f"{prefix}_angvel_{axis}")
        names.extend(
            [
                f"{prefix}_boost",
                f"{prefix}_on_ground",
                f"{prefix}_has_flip",
                f"{prefix}_has_jumped",
                f"{prefix}_is_supersonic",
                f"{prefix}_air_time",
                f"{prefix}_flip_time",
                f"{prefix}_demo_timer",
                f"{prefix}_is_orange",
            ]
        )
    if config.include_boost_pads:
        for i in range(6):
            names.append(f"boost_pad_large_{i}_timer")
        names.append("boost_pad_small_ready_frac")
    return names


# ---------------------------------------------------------------------------
# Universal extractor — handles 1v1 / 2v2 / 3v3 in a single 133-dim vector.
# ---------------------------------------------------------------------------

_N_CARS_MAX_UNIVERSAL = 6       # 3 blue slots + 3 orange slots
_CAR_FEATURES = 18
_BALL_FEATURES = 9
_BOOST_SUMMARY_FEATURES = 7

# 9 (ball) + 6*18 (car slots) + 7 (boost) + 6 (masks) + 3 (n_active)
_UNIVERSAL_SIZE = (
    _BALL_FEATURES
    + _N_CARS_MAX_UNIVERSAL * _CAR_FEATURES
    + _BOOST_SUMMARY_FEATURES
    + _N_CARS_MAX_UNIVERSAL   # per-slot presence mask
    + 3                        # n_blue, n_orange, n_total (normalized)
)


def universal_feature_size() -> int:
    """Return the fixed output dimension of extract_universal_frame_features.

    Layout: 9 (ball) + 108 (6 car slots × 18) + 7 (boost summary)
            + 6 (slot masks) + 3 (n_active scalars) = 133.
    """
    return _UNIVERSAL_SIZE


def universal_feature_names() -> list[str]:
    """Human-readable names for all 133 dimensions of the universal vector."""
    names: list[str] = []

    # --- ball (9) ---
    for axis in ("x", "y", "z"):
        names.append(f"ball_pos_{axis}")
    for axis in ("x", "y", "z"):
        names.append(f"ball_linvel_{axis}")
    for axis in ("x", "y", "z"):
        names.append(f"ball_angvel_{axis}")

    # --- car slots (6 × 18) ---
    for slot in range(_N_CARS_MAX_UNIVERSAL):
        team = "blue" if slot < 3 else "orange"
        idx_in_team = slot if team == "blue" else slot - 3
        prefix = f"car_{team}{idx_in_team}"
        for axis in ("x", "y", "z"):
            names.append(f"{prefix}_pos_{axis}")
        for axis in ("x", "y", "z"):
            names.append(f"{prefix}_linvel_{axis}")
        for axis in ("x", "y", "z"):
            names.append(f"{prefix}_angvel_{axis}")
        names.extend(
            [
                f"{prefix}_boost",
                f"{prefix}_on_ground",
                f"{prefix}_has_flip",
                f"{prefix}_has_jumped",
                f"{prefix}_is_supersonic",
                f"{prefix}_air_time",
                f"{prefix}_flip_time",
                f"{prefix}_demo_timer",
                f"{prefix}_is_orange",
            ]
        )

    # --- boost summary (7) ---
    for i in range(6):
        names.append(f"boost_pad_large_{i}_timer")
    names.append("boost_pad_small_ready_frac")

    # --- slot masks (6) ---
    for slot in range(_N_CARS_MAX_UNIVERSAL):
        team = "blue" if slot < 3 else "orange"
        idx_in_team = slot if team == "blue" else slot - 3
        names.append(f"mask_car_{team}{idx_in_team}")

    # --- n_active scalars (3) ---
    names.extend(["n_blue_norm", "n_orange_norm", "n_total_norm"])

    assert len(names) == _UNIVERSAL_SIZE, (
        f"universal_feature_names length {len(names)} != {_UNIVERSAL_SIZE}"
    )
    return names


def extract_universal_frame_features(
    state: GameState,
    *,
    n_cars_max: int = _N_CARS_MAX_UNIVERSAL,
    inverted: bool = False,
) -> np.ndarray:
    """Build a 133-dim feature vector that works for any game mode (1v1/2v2/3v3).

    Layout (in order):
        [0:9]      ball physics (position + linear vel + angular vel, normalized)
        [9:117]    6 car slots × 18 features each:
                     slots 0-2  → blue cars  (sorted by AgentID, zero-padded if absent)
                     slots 3-5  → orange cars (sorted by AgentID, zero-padded if absent)
        [117:124]  boost pad summary (6 large timers + small_ready fraction)
        [124:130]  car presence mask — 1.0 if the slot has a real car, 0.0 otherwise
        [130:133]  n_active_cars as normalized scalars:
                     [n_blue/3, n_orange/3, n_total/6]

    Args:
        state:       rlgym-compat GameState for a single frame.
        n_cars_max:  Maximum slots (default 6 for 3v3; do not change for 1v1/2v2 —
                     use padding so the output dimension remains constant at 133).
        inverted:    If True, use the orange-perspective (inverted) physics view.

    Returns:
        np.ndarray of shape (133,) and dtype float32.
    """
    n_slots_per_team = n_cars_max // 2

    # Sort by AgentID (stable dict key) — see extract_frame_features for the bug
    # rationale (id(c) memory-address sort scrambled per-car identity per frame).
    cars_blue = [
        c for _, c in sorted(
            ((aid, c) for aid, c in state.cars.items() if c.team_num == BLUE_TEAM),
            key=lambda kv: str(kv[0]),
        )
    ]
    cars_orange = [
        c for _, c in sorted(
            ((aid, c) for aid, c in state.cars.items() if c.team_num == ORANGE_TEAM),
            key=lambda kv: str(kv[0]),
        )
    ]

    n_blue = len(cars_blue)
    n_orange = len(cars_orange)

    # Pad to n_slots_per_team slots each team
    blue_slots = (cars_blue + [None] * n_slots_per_team)[:n_slots_per_team]
    orange_slots = (cars_orange + [None] * n_slots_per_team)[:n_slots_per_team]
    all_slots = blue_slots + orange_slots   # len == n_cars_max

    zero_car = np.zeros(_CAR_FEATURES, dtype=np.float32)

    parts: list[np.ndarray] = []

    # ball (9)
    parts.append(ball_features(state, inverted=inverted))

    # car slots (n_cars_max × 18)
    for car in all_slots:
        parts.append(zero_car if car is None else car_features(car, inverted=inverted))

    # boost summary (7)
    parts.append(boost_pad_summary(state, inverted=inverted))

    # slot masks (n_cars_max)
    mask = np.array(
        [1.0 if car is not None else 0.0 for car in all_slots],
        dtype=np.float32,
    )
    parts.append(mask)

    # n_active scalars (3): normalized to [0,1]
    parts.append(
        np.array(
            [n_blue / n_slots_per_team, n_orange / n_slots_per_team, (n_blue + n_orange) / n_cars_max],
            dtype=np.float32,
        )
    )

    out = np.concatenate(parts).astype(np.float32)
    assert out.shape == (_UNIVERSAL_SIZE,), (
        f"extract_universal_frame_features: expected shape ({_UNIVERSAL_SIZE},), got {out.shape}"
    )
    return out
