"""Teleop: the motion rules on a fake pair, the simulated pair, the SDK, monorobot start, and the built package."""

from __future__ import annotations

import itertools
import json
import logging
import math
import os
import signal
import time
from pathlib import Path

import pytest
from conftest import (
    MID_DEG,
    REST_DEG,
    FakeArm,
    FakeLeader,
    LineReader,
    assert_motion_ok,
    built_manifest,
    envelope_deg,
    make_app,
    run_paced,
    run_ticks,
    run_under_sdk,
    sim_joints_deg,
    sim_pair_copy,
    start_on_sim_pair,
    urdf_limits_deg,
)

VERSION = "0.1.0"
WHATS_NEW = "First version."
RATE_HZ = 60
FOLLOW_DEG_PER_S = 3.0 * RATE_HZ               # D37: following, at most 3 degrees a tick
PHASES = {"glide": 30.0, "follow": FOLLOW_DEG_PER_S}
GRIPPER_STEPS = {"follow": 0.1}
MISSING_LEADER = "Teleop needs the SO-101 leader. Plug it in, then run again."
HELLO_DEG = (0.0, -40.0, 45.0, -15.0, 0.0)


def run_with(main, arm: FakeArm, leader: FakeLeader, ticks: int, move=None, robot_type: str = "hf_so101_follower"):
    """Teleop on ``arm`` with ``leader`` for ``ticks`` ticks; ``move(app)`` moves the leader before each tick."""
    app = make_app(main, arm, robot_type, peers={"leader": leader})
    run_ticks(app, ticks, before_tick=move)
    return app


def hand(path):
    """A ``before_tick`` that puts the leader where ``path(seconds)`` says, each tick."""

    def move(app) -> None:
        joints, opening = path(app.tick / RATE_HZ)
        leader = app.peers["leader"]
        leader.joints_deg, leader.opening = list(joints), opening

    return move


def slow_path(t: float):
    """A hand moving the leader around a pose in front of the base, as the simulated leader moves."""
    mid = (0.0, -40.0, 20.0, 20.0, 0.0)
    joints = [m + 20.0 * math.sin(2 * math.pi * t / 8.0 + i * math.pi / 3) for i, m in enumerate(mid)]
    return joints, 0.5 + 0.3 * math.sin(2 * math.pi * t / 8.0 + math.pi)


def flick_path(t: float):
    """Still, then a sudden 40 degree flick on three joints (a quick hand, or a bad reading)."""
    if t < 4.0:
        return list(HELLO_DEG), 0.2
    return [HELLO_DEG[0] + 40.0, HELLO_DEG[1] - 40.0, HELLO_DEG[2] + 40.0, *HELLO_DEG[3:]], 1.0


def far_path(t: float):
    """The leader moved past the follower's envelope on every joint."""
    return [90.0, -100.0, 96.0, -95.0, 160.0], 1.0


PATHS = {"still": lambda t: (list(HELLO_DEG), 0.5), "slow": slow_path, "flick": flick_path, "far": far_path}
STARTS = {"rest": REST_DEG, "mid": MID_DEG}


def more_ticks(app, n: int) -> None:
    """``loop()`` ``n`` more times, after ``run_ticks``, with no new ``setup()``."""
    for _ in range(n):
        app.loop()
        app.tick += 1


def phase_runs(arm: FakeArm) -> list[tuple[str, int]]:
    """The phases the targets were sent in, in order, with how many targets each."""
    runs: list[tuple[str, int]] = []
    for label in arm.labels():
        if runs and runs[-1][0] == label:
            runs[-1] = (label, runs[-1][1] + 1)
        else:
            runs.append((label, 1))
    return runs


# --------------------------------------------------------------------------- setup


def test_needs_the_leader(main):
    app = make_app(main, FakeArm(), peers={})
    with pytest.raises(RuntimeError) as exc:
        app.setup()
    assert str(exc.value) == MISSING_LEADER


def test_needs_the_leader_on_an_engine_that_sets_no_peers(main):
    app = make_app(main, FakeArm())
    del app.peers
    with pytest.raises(RuntimeError) as exc:
        app.setup()
    assert str(exc.value) == MISSING_LEADER


def test_needs_an_so101(main):
    app = make_app(main, FakeArm(), robot_type="lynxmotion_al5d", peers={"leader": FakeLeader()})
    with pytest.raises(RuntimeError) as exc:
        app.setup()
    assert str(exc.value) == "Teleop runs on the SO-101 arm."


def test_nothing_moves_in_setup(main):
    arm, leader = FakeArm(REST_DEG), FakeLeader()
    app = make_app(main, arm, peers={"leader": leader})
    app.setup()
    assert arm.targets == []
    assert leader.reads == 0
    assert arm.reads["get_joint_positions"] == 1


# --------------------------------------------------------------------------- motion


@pytest.mark.parametrize("path", sorted(PATHS))
@pytest.mark.parametrize("start", sorted(STARTS))
def test_motion_follows_the_rules(main, start, path):
    arm = FakeArm(STARTS[start], gripper=0.1)
    run_with(main, arm, FakeLeader(), 12 * RATE_HZ, move=hand(PATHS[path]))
    assert_motion_ok(arm, RATE_HZ, PHASES, gripper_steps=GRIPPER_STEPS)
    assert {method for method, *_ in arm.targets} == {"send_teleop_action"}
    assert [name for name, _ in phase_runs(arm)] == ["glide", "follow"]


@pytest.mark.parametrize("gap_deg", [150.0, 60.0])
def test_the_glide_takes_the_gap_over_30_degrees_a_second(main, gap_deg):
    """D37: the glide is speed based. A 150 degree gap takes 5 s, on a straight line, and ends on the leader."""
    start = [*HELLO_DEG[:4], -gap_deg / 2]
    goal = [*HELLO_DEG[:4], gap_deg / 2]
    arm = FakeArm(start)
    app = run_with(main, arm, FakeLeader(goal), round(gap_deg / 30.0 * RATE_HZ) + 30)
    glide = arm.sent("glide")
    assert len(glide) / RATE_HZ == pytest.approx(gap_deg / 30.0, abs=0.2)
    steps = [abs(b[4] - a[4]) for a, b in zip([start, *glide], glide)]
    assert max(steps) == pytest.approx(30.0 / RATE_HZ)
    for pose in glide:                                   # only the joint that has a gap moves
        assert pose[:4] == pytest.approx(list(HELLO_DEG[:4]))
    assert glide[-1] == pytest.approx(goal, abs=1e-9)    # the glide itself ends on the leader
    assert arm.sent("follow")[0] == pytest.approx(goal, abs=1e-9)
    assert app.phase == "follow"
    # The leader is still: no tick, the first follow tick included, moves faster than the glide.
    assert_motion_ok(arm, RATE_HZ, {"glide": 30.0, "follow": 30.0})


@pytest.mark.parametrize("gap_deg", [0.0, 2.5, 10.0, 29.0])
def test_a_small_gap_glides_for_at_least_a_second(main, gap_deg):
    """D37: at least 1 s. A gap under 30 degrees glides at the gap a second, for 1 s, and ends on the leader."""
    start = list(HELLO_DEG)
    goal = [HELLO_DEG[0] + gap_deg, *HELLO_DEG[1:]]
    arm = FakeArm(start)
    run_with(main, arm, FakeLeader(goal), 2 * RATE_HZ)
    glide = arm.sent("glide")
    assert len(glide) == RATE_HZ                         # 1 s, from the first glide tick
    assert glide[0][0] - start[0] == pytest.approx(gap_deg / RATE_HZ)
    assert glide[-1] == pytest.approx(goal, abs=1e-9)
    assert arm.sent("follow")[0] == pytest.approx(goal, abs=1e-9)   # no step left for the first follow tick
    steps = [b[0] - a[0] for a, b in itertools.pairwise([start, *arm.sent()])]
    assert max(steps) == pytest.approx(gap_deg / RATE_HZ)
    assert_motion_ok(arm, RATE_HZ, {"glide": 30.0, "follow": 30.0})


def test_a_glide_from_rest_moves_straight_toward_the_envelope(main):
    """From the folded rest pose (outside the envelope) every joint moves in, never further out."""
    arm = FakeArm(REST_DEG)
    run_with(main, arm, FakeLeader(REST_DEG), 3 * RATE_HZ)
    edge = main.clip_to_envelope(REST_DEG)
    assert arm.sent()[-1] == pytest.approx(edge)
    assert_motion_ok(arm, RATE_HZ, PHASES, gripper_steps=GRIPPER_STEPS)


def test_the_glide_catches_a_leader_that_moves_away(main):
    arm = FakeArm(MID_DEG)
    leader = FakeLeader([MID_DEG[0] + 3.0, *MID_DEG[1:]])

    def move(app) -> None:                      # moves away at 15 degrees a second for 2 s
        if app.tick < 2 * RATE_HZ:
            leader.joints_deg[0] += 15.0 / RATE_HZ

    app = make_app(main, arm, peers={"leader": leader})
    run_ticks(app, 8 * RATE_HZ, before_tick=move)
    assert app.phase == "follow"
    assert arm.sent()[-1] == pytest.approx(leader.joints_deg)
    assert_motion_ok(arm, RATE_HZ, PHASES, gripper_steps=GRIPPER_STEPS)


def test_follow_copies_the_leader_3_degrees_a_tick_at_most(main):
    arm = FakeArm(MID_DEG)
    leader = FakeLeader(MID_DEG)
    app = make_app(main, arm, peers={"leader": leader})
    run_ticks(app, RATE_HZ + 1)                 # the glide lasts 1 s even with no gap
    assert app.phase == "follow"
    leader.joints_deg[2] += 20.0                # the hand moves the elbow 20 degrees at once
    before = len(arm.targets)
    more_ticks(app, 10)
    elbow = [joints[2] for joints in arm.sent()[before - 1:]]
    assert [round(b - a, 9) for a, b in itertools.pairwise(elbow)] == [3.0] * 6 + [2.0] + [0.0] * 3
    assert arm.sent()[-1] == pytest.approx(leader.joints_deg)


def test_targets_stay_in_the_envelope(main):
    arm = FakeArm(MID_DEG)
    run_with(main, arm, FakeLeader(), 10 * RATE_HZ, move=hand(far_path))
    last = arm.sent()[-1]
    assert last == pytest.approx(main.clip_to_envelope(far_path(0)[0]))
    corner = [hi if v > 0 else lo for v, (lo, hi) in zip(far_path(0)[0], envelope_deg())]
    assert all(abs(v - c) <= 1.0 for v, c in zip(last, corner)), (last, corner)
    assert_motion_ok(arm, RATE_HZ, PHASES, gripper_steps=GRIPPER_STEPS)


def test_the_app_envelope_sits_inside_the_urdf_one(main):
    for name, (lo, hi), (ulo, uhi) in zip(main.JOINTS, (main.ENVELOPE_DEG[j] for j in main.JOINTS), envelope_deg()):
        assert ulo - 1e-9 <= lo <= ulo + 1.0, name
        assert uhi - 1.0 <= hi <= uhi + 1e-9, name


def test_the_motion_check_catches_a_target_past_the_urdf_limits():
    """The SDK's safety check has no SO-101 limits, so ``assert_motion_ok`` checks the URDF ones
    itself: a small step inward that is still past the shoulder's limit fails."""
    lowest = urdf_limits_deg()[1][0]
    arm = FakeArm([0.0, lowest - 6.0, 30.0, 0.0, 0.0])
    arm.send_teleop_action([math.radians(v) for v in (0.0, lowest - 5.5, 30.0, 0.0, 0.0)], 0.0)
    with pytest.raises(AssertionError, match="URDF"):
        assert_motion_ok(arm, RATE_HZ, PHASES)


def test_the_tests_never_see_the_real_home(tmp_path):
    """CONTRACTS 0.5: HOME and MONOROBOT_HOME both point under the test's temp folder."""
    assert Path.home().resolve().is_relative_to(tmp_path.resolve())
    assert Path(os.environ["MONOROBOT_HOME"]).resolve().is_relative_to(tmp_path.resolve())


def test_the_gripper_ramps_then_follows(main):
    arm = FakeArm(MID_DEG, gripper=0.0)
    leader = FakeLeader([MID_DEG[0] + 30.0, *MID_DEG[1:]], opening=0.9)
    app = make_app(main, arm, peers={"leader": leader})
    run_ticks(app, 3 * RATE_HZ)
    glide = [g for _, _, g, label in arm.targets if label == "glide"]
    assert glide[-1] == pytest.approx(min(0.9, len(glide) / RATE_HZ))     # 1 a second at most
    leader.opening = 0.0
    more_ticks(app, 20)
    assert arm.targets[-1][2] == pytest.approx(0.0)
    assert_motion_ok(arm, RATE_HZ, PHASES, gripper_steps=GRIPPER_STEPS)


def test_a_gripper_reading_past_open_is_clamped(main):
    arm = FakeArm(MID_DEG, gripper=1.0)
    arm.gripper = 1.02
    leader = FakeLeader(MID_DEG, opening=1.05)
    run_with(main, arm, leader, RATE_HZ)
    assert all(0.0 <= g <= 1.0 for _, _, g, _ in arm.targets)


def test_reads_the_leader_once_a_tick_and_sends_one_target(main):
    """One read and one target per tick, no blocking move: a Stop lands between any two ticks."""
    arm, leader = FakeArm(REST_DEG), FakeLeader()
    app = run_with(main, arm, leader, 150)
    assert leader.reads == 150
    assert len(arm.targets) == 150
    assert arm.reads == {"get_joint_positions": 1, "gripper_get_width": 1, "get_telemetry": 0}
    assert app.finish_calls == []


def test_a_leader_in_the_app_process_works(main):
    """An in-process leader gives a TelemetryFrame object, not a dict."""
    arm = FakeArm(MID_DEG)
    leader = FakeLeader([MID_DEG[0] + 10.0, *MID_DEG[1:]], as_frame=True)
    run_with(main, arm, leader, 3 * RATE_HZ)
    assert arm.sent()[-1] == pytest.approx(leader.joints_deg)


def test_an_unusable_reading_is_skipped(main, caplog):
    caplog.set_level(logging.INFO, logger="teleop")
    arm = FakeArm(MID_DEG)
    leader = FakeLeader(MID_DEG)
    app = make_app(main, arm, peers={"leader": leader})
    run_ticks(app, 10)
    leader.joints_deg[1] = float("nan")
    more_ticks(app, 3)
    assert len(arm.targets) == 10                  # nothing sent while the reading is bad
    leader.joints_deg[1] = MID_DEG[1]
    more_ticks(app, 3)
    assert len(arm.targets) == 13
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings == ["The leader sent a reading that isn't usable, so the follower waits for the next one."]


def test_logs_are_short_and_plain(main, caplog):
    caplog.set_level(logging.INFO, logger="teleop")
    arm = FakeArm(MID_DEG)
    app = run_with(main, arm, FakeLeader([10.0, -20.0, 30.0, 23.4, 5.0]), 11 * RATE_HZ)
    app.on_shutdown()
    lines = [r.getMessage() for r in caplog.records if r.name == "teleop"]
    assert lines == ["Reading the leader.", "Ready: move the leader and the follower copies it.",
                     "Following, wrist at 23 degrees.", "Following, wrist at 23 degrees.", "Stopped."]
    for line in lines:
        assert line[0].isupper() and line.endswith(".") and "\u2014" not in line
        assert "so101" not in line.lower() and "monorobot" not in line.lower()


# --------------------------------------------------------------------------- the simulated pair


def folded(state: dict) -> bool:
    """The simulated follower's state file says it folded to rest and let go.

    The fold moves the shoulder, elbow and wrist to the rest pose; the base and the wrist roll stay
    where the leader left them.
    """
    return state["torque"] == "off" and state["joints_deg"][1:4] == pytest.approx(list(REST_DEG[1:4]), abs=0.5)


def test_on_the_simulated_pair(main, sim_follower, sim_leader):
    """After the glide and 3 s of following, the follower is within 5 degrees of the leader on every joint."""
    assert sim_joints_deg(sim_follower) == pytest.approx(list(REST_DEG), abs=0.01)
    app = make_app(main, sim_follower, "sim_so101_follower", peers={"leader": sim_leader})
    following = {"ticks": 0}

    def until(app) -> bool:
        if app.phase == "follow":
            following["ticks"] += 1
        return following["ticks"] >= 3 * RATE_HZ

    run_paced(app, 30 * RATE_HZ, until=until)
    assert app.phase == "follow" and following["ticks"] == 3 * RATE_HZ
    follower, leader = sim_joints_deg(sim_follower), sim_joints_deg(sim_leader)
    assert follower == pytest.approx(leader, abs=5.0)
    assert app.peers["leader"] is sim_leader


def test_the_sdk_builds_the_leader_and_folds_the_follower_at_the_end(main, tmp_path, caplog):
    """The SDK's own run(): the leader comes from robot.yaml's also_uses; at the end the follower folds to rest."""
    caplog.set_level(logging.INFO, logger="teleop")
    app, code, state = run_under_sdk(main, tmp_path, until=lambda app: app.followed >= 2 * RATE_HZ)
    assert code == 0 and app.error is None
    assert set(app.peers) == {"leader"}
    assert type(app.peers["leader"]).__name__ == "SimSO101LeaderAdapter"
    assert not app.peers["leader"].is_connected()               # the framework closed it
    assert folded(state)
    lines = [r.getMessage() for r in caplog.records if r.name == "teleop"]
    assert lines == ["Reading the leader.", "Ready: move the leader and the follower copies it.", "Stopped."]


def test_the_sdk_does_not_start_without_the_leader(main, tmp_path):
    """The leader is required: unplugged, the run doesn't start, and the follower never moves."""
    from monorobot.hal.adapters.sim_so101._state import update_state

    update_state(tmp_path / "sim" / "sim-leader-1.json", {"connected": False})
    with pytest.raises(RuntimeError, match="connect"):
        run_under_sdk(main, tmp_path)
    state = json.loads((tmp_path / "sim" / "sim-follower-1.json").read_text())
    assert state["joints_deg"] == pytest.approx(list(REST_DEG), abs=0.01)


@pytest.mark.slow
def test_monorobot_start_on_the_simulated_pair(tmp_path):
    """``monorobot start`` with the simulated pair in robot.yaml: it follows the leader for 10 s, then
    one Ctrl+C (SIGINT) ends it and the follower folds to rest."""
    copy, binding, sim = sim_pair_copy(tmp_path)
    proc = start_on_sim_pair(copy, binding, tmp_path)
    out = LineReader(proc)
    try:
        assert out.wait_for("Ready: move the leader and the follower copies it.", 60), out.text()
        ready_at = time.monotonic()
        moved = 0.0
        while time.monotonic() - ready_at < 10.5:
            time.sleep(0.5)
            follower = json.loads((sim / "sim-follower-1.json").read_text())["joints_deg"]
            moved = max(moved, max(abs(a - b) for a, b in zip(follower, REST_DEG)))
            assert proc.poll() is None, out.text()
        proc.send_signal(signal.SIGINT)
        code = proc.wait(timeout=60)
    except BaseException:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)
        raise
    text = out.text()
    assert code == 0, text
    assert "Using the sim_so101_follower settings in" in text, text   # the binding reached the follower
    assert moved > 20.0, text                                      # it followed the leader off rest
    assert text.count("Following, wrist at") >= 1, text
    assert "Stopped." in text
    assert "single goal" not in text and "was cut to" not in text
    assert folded(json.loads((sim / "sim-follower-1.json").read_text()))


def test_start_never_runs_on_a_real_robot_here(tmp_path):
    """The copy ``monorobot start`` runs names only simulated robots, and no serial port."""
    copy, binding, _ = sim_pair_copy(tmp_path)
    for text in ((copy / "robot.yaml").read_text(), binding.read_text()):
        assert "hf_so101" not in text and "/dev/" not in text


# --------------------------------------------------------------------------- the package


def test_built_manifest(tmp_path):
    manifest, names = built_manifest(tmp_path)
    assert manifest["schema"] == "monorobot/1.2"
    assert manifest["name"] == "teleop"
    assert manifest["version"] == VERSION
    assert manifest["author"] == {"handle": "grand-wizard-create"}
    assert manifest["compat"]["robots"] == ["hf_so101_follower"]
    assert manifest["also_uses"] == [{"name": "leader", "type": "hf_so101_leader", "required": True}]
    assert manifest["devices"] == {"required": [], "optional": []}
    assert manifest["permissions"]["network"] is False
    assert manifest["permissions"]["filesystem_write"] == []
    assert manifest["license"] == "Apache-2.0"
    assert manifest["whats_new"] == WHATS_NEW
    assert manifest["description"] == "Move the leader arm and the follower copies it."
    assert sorted(names) == [".gitignore", "README.md", "main.py", "manifest.json", "pyproject.toml", "robot.yaml"]


def test_the_readme_puts_the_leader_port_in_robot_yaml():
    """A robot binding reaches only the main robot for now, so the terminal steps never bind the
    leader: its port goes in robot.yaml, next to the leader's own kwargs."""
    import re

    import yaml
    from conftest import APP_DIR

    text = (APP_DIR / "README.md").read_text(encoding="utf-8")
    assert "robot bind hf_so101_leader" not in text
    leader, = yaml.safe_load((APP_DIR / "robot.yaml").read_text())["also_uses"]
    snippet = re.search(r"```yaml\n(.*?)```", text, re.DOTALL)
    assert snippet is not None
    assert yaml.safe_load(snippet.group(1))["kwargs"] == {**leader["kwargs"], "port": "<leader port>"}


def test_no_uv_lock_and_no_em_dashes():
    """T2: no uv.lock to be packed by mistake; the copy rules: no em dash in any file of the app."""
    from conftest import APP_DIR

    assert not (APP_DIR / "uv.lock").exists()
    def kept(path) -> bool:
        """A file of the app: not in a hidden folder (git's, a tool's cache), a build or a bytecode folder."""
        folders = path.relative_to(APP_DIR).parts[:-1]
        return not any(part.startswith(".") or part in {"__pycache__", "dist"} for part in folders)

    files = [p for p in APP_DIR.rglob("*") if p.is_file() and kept(p)]
    assert APP_DIR / "main.py" in files
    for path in files:
        if path.name != "LICENSE":
            assert "\u2014" not in path.read_text(encoding="utf-8"), path
