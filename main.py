"""Teleop: move the leader arm and the follower copies it.

Runs on the SO-101 follower arm with the SO-101 leader arm (``also_uses`` in ``robot.yaml``).
Every move is streamed, one small step per tick, so no joint ever jumps. First the follower
glides onto the leader, at most 30 degrees a second on the joint with the farthest to go (a 150
degree gap takes 5 s, a small one at least 1 s). Then it copies the leader, each joint target at
most 3 degrees from the one before. Targets stay 10 degrees inside the follower's limits, and the
base turns at most 60 degrees either way: a leader moved past them is copied up to the edge.

The work runs as phases in ``loop()`` (``glide``, then ``follow``), so a Stop lands on any tick.
It runs until stopped; then the follower folds back to rest on its own.
"""

from __future__ import annotations

import logging
import math
import sys
from collections.abc import Mapping, Sequence
from typing import Any

from monorobot.sdk.app import App

log = logging.getLogger("teleop")

SO101 = frozenset({"hf_so101_follower", "sim_so101_follower"})
JOINTS: tuple[str, ...] = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")

GLIDE_DEG_PER_S = 30.0       # onto the leader, on the joint with the farthest to go
GLIDE_MIN_S = 1.0            # a glide onto the leader lasts at least this long
FOLLOW_STEP_DEG = 3.0        # while following, no joint target moves more than this in one tick
GRIPPER_GLIDE_PER_S = 1.0    # gripper opening per second while gliding: closed to open in 1 s
GRIPPER_STEP = 0.1           # while following, the most the gripper target moves in one tick
REPORT_EVERY_S = 5.0

# Every target stays 10 degrees inside the follower's URDF limits, and the base turns at most 60
# degrees either way.
ENVELOPE_DEG: dict[str, tuple[float, float]] = {
    "shoulder_pan": (-60.0, 60.0),
    "shoulder_lift": (-90.0, 90.0),
    "elbow_flex": (-86.8, 86.8),
    "wrist_flex": (-85.0, 85.0),
    "wrist_roll": (-147.0, 152.0),
}


def clip_to_envelope(joints_deg: Sequence[float]) -> list[float]:
    """The pose held inside ``ENVELOPE_DEG``, joint by joint."""
    return [min(max(float(v), ENVELOPE_DEG[name][0]), ENVELOPE_DEG[name][1]) for name, v in zip(JOINTS, joints_deg)]


def clamp01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def step_toward(last: float, aim: float, most: float) -> float:
    """``aim``, or ``last`` moved toward it by ``most`` when it is further than that."""
    return aim if abs(aim - last) <= most else last + math.copysign(most, aim - last)


def leader_pose(leader: Any) -> tuple[list[float], float] | None:
    """One read of the leader: its arm joints in degrees and its gripper opening (0 closed, 1 open).

    Telemetry gives both from one read of the arm; it counts the gripper the other way round (0
    open). A leader in its own process sends telemetry as a dict, one in the app's process as a
    ``TelemetryFrame``. None when the reading isn't usable (not numbers, or too few joints).
    """
    frame = leader.get_telemetry()
    joints = frame["joints"] if isinstance(frame, Mapping) else frame.joints
    closed = frame["gripper"] if isinstance(frame, Mapping) else frame.gripper
    try:
        values = [math.degrees(float(v)) for v in joints[: len(JOINTS)]]
        opening = 1.0 - float(closed)
    except (TypeError, ValueError):
        return None
    if len(values) < len(JOINTS) or not all(math.isfinite(v) for v in [*values, opening]):
        return None
    return values, clamp01(opening)


class Mirror:
    """The follower's targets, one per tick: a glide onto the leader, then a copy of it.

    ``phase`` is ``glide`` until a glide tick, at least ``GLIDE_MIN_S`` in, puts the follower on
    the leader's pose itself; then ``follow``, so the first follow tick has no gap left to close.
    Each glide tick moves on a straight line toward where the leader is now: the joint with the
    farthest to go moves at most ``GLIDE_DEG_PER_S`` a second, and no more than its gap shared
    over the ticks left of ``GLIDE_MIN_S``. So a big gap takes the gap over ``GLIDE_DEG_PER_S``,
    a small one ``GLIDE_MIN_S``, and a leader that moves away is still caught. Following, each
    joint target is at most ``FOLLOW_STEP_DEG`` from the one before. Targets aim at the
    leader's pose held inside the envelope.
    """

    def __init__(self, start_deg: Sequence[float], start_gripper: float, rate_hz: float) -> None:
        self.rate_hz = float(rate_hz)
        self.joints = [float(v) for v in start_deg]
        self.gripper = clamp01(start_gripper)
        self.phase = "glide"
        self.glided = 0                         # glide ticks so far
        self.min_ticks = max(1, math.ceil(GLIDE_MIN_S * self.rate_hz - 1e-9))
        self._on_leader = False                 # the last glide tick ended on the leader

    def step(self, leader_deg: Sequence[float], leader_gripper: float) -> tuple[list[float], float]:
        """The next target, ``(joints_deg, gripper)``, toward the leader as read this tick."""
        aim = clip_to_envelope(leader_deg)
        aim_gripper = clamp01(leader_gripper)
        if self.phase == "glide" and self._on_leader:
            self.phase = "follow"
        if self.phase == "glide":
            self._glide(aim)
            self.gripper = step_toward(self.gripper, aim_gripper, GRIPPER_GLIDE_PER_S / self.rate_hz)
        else:
            self.joints = [step_toward(j, a, FOLLOW_STEP_DEG) for a, j in zip(aim, self.joints)]
            self.gripper = step_toward(self.gripper, aim_gripper, GRIPPER_STEP)
        return list(self.joints), self.gripper

    def _glide(self, aim: list[float]) -> None:
        """One glide tick toward ``aim``."""
        self.glided += 1
        ticks_left = max(1, self.min_ticks - self.glided + 1)
        gap = max(abs(a - j) for a, j in zip(aim, self.joints))
        most = min(GLIDE_DEG_PER_S / self.rate_hz, gap / ticks_left)
        if gap <= most:
            self.joints = list(aim)
        else:
            share = most / gap
            self.joints = [j + (a - j) * share for a, j in zip(aim, self.joints)]
        self._on_leader = self.glided >= self.min_ticks and self.joints == aim


class Teleop(App):
    def setup(self) -> None:
        if self.config.robot.type not in SO101:
            raise RuntimeError("Teleop runs on the SO-101 arm.")
        self.leader = getattr(self, "peers", {}).get("leader")
        if self.leader is None:
            raise RuntimeError("Teleop needs the SO-101 leader. Plug it in, then run again.")
        start = [math.degrees(v) for v in self.robot.get_joint_positions()]
        self.mirror = Mirror(start, self.robot.gripper_get_width(), self.rate_hz)
        self.followed = 0          # ticks spent following
        self.skipped = 0           # leader readings that weren't usable
        log.info("Reading the leader.")

    @property
    def phase(self) -> str:
        mirror = getattr(self, "mirror", None)
        return mirror.phase if mirror is not None else "setup"

    def loop(self) -> None:
        pose = leader_pose(self.leader)
        if pose is None:
            # The follower keeps its last target this tick.
            if self.skipped == 0:
                log.warning("The leader sent a reading that isn't usable, so the follower waits for the next one.")
            self.skipped += 1
            return
        before = self.mirror.phase
        joints, gripper = self.mirror.step(*pose)
        if before == "glide" and self.mirror.phase == "follow":
            log.info("Ready: move the leader and the follower copies it.")
        self.robot.send_teleop_action([math.radians(v) for v in joints], gripper)
        if self.mirror.phase == "follow":
            self.followed += 1
            if self.followed % max(1, round(REPORT_EVERY_S * self.rate_hz)) == 0:
                log.info("Following, wrist at %d degrees.", round(joints[JOINTS.index("wrist_flex")]))

    def on_shutdown(self) -> None:
        # The framework closes the leader; the app never does.
        if getattr(self, "mirror", None) is not None:
            log.info("Stopped.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    sys.exit(Teleop.from_config("robot.yaml").run())
