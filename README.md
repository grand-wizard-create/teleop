# Teleop

Move the leader arm by hand and the follower arm copies it, joint by joint, with its gripper. It
runs until you stop it.

## What it runs on

- The SO-101 follower arm, or the simulated SO-101 follower.
- The SO-101 leader arm, or the simulated SO-101 leader. Teleop needs it.

Every move is smooth: the follower never jumps. When it starts, it glides onto the leader at no
more than 30 degrees a second, so a big gap takes a few seconds. Then it copies the leader, a few
degrees at a time. It keeps a little inside the arm's limits: move the leader past them and the
follower stops at the edge. When you stop the app, the follower folds back to rest.

## Run it from Robot App Store

Find Teleop in Discover, install it, choose your follower arm, and click Run. Plug in the leader
arm first. Wait for "Ready" in the log, then move the leader.

## Run it from a terminal

```
monorobot pull grand-wizard-create/teleop
monorobot robot bind hf_so101_follower --kwarg port=<port>
```

A robot binding reaches only the follower for now, so the leader's port goes in the app's own
`robot.yaml`. Open `~/.monorobot/apps/teleop/robot.yaml` and add the port to the leader's
`kwargs`, under `also_uses`:

```yaml
    kwargs: {use_default_calibration: true, port: <leader port>}
```

Then start it:

```
monorobot start teleop
```

Replace `<port>` and `<leader port>` with the USB ports of your two arms.

## Stop it

In Robot App Store, press Space or Esc. In a terminal, press Ctrl+C: the follower folds back to
rest. Press Ctrl+C a second time to stop the arm where it is.

## License

Apache License 2.0. See `LICENSE`.
