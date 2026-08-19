#!/usr/bin/env python3
"""Exercise cmd_verify's reporting path with no camera and no robot.

The two previous real runs both crashed in the teardown/summary region, so the
patched version of it has never actually executed. Stub the hardware, replay
synthetic readings through the real cmd_verify, and check the verdict comes out
and matches the data.

Three scenarios, because the interesting failure is the third one:
  good      -- board steady, arm moved a lot        -> PASS
  swimming  -- board drifts with the arm            -> FAIL
  parked    -- board steady, arm barely moved       -> INCONCLUSIVE
"""
import json
import sys
import types

import numpy as np

import handeye_calib as H


class FakeFeed:
    """Stands in for the 30004 RT client. Walks the flange through `poses`."""

    def __init__(self, poses):
        self.poses = poses
        self.i = 0

    def start(self):
        pass

    def stop(self):
        pass

    def latest(self):
        p = self.poses[min(self.i, len(self.poses) - 1)]
        self.i += 1
        return {'tool': p, 'qd': np.zeros(6)}      # qd=0 -> always "still"


def run(name, tool_poses, board_base, jitter_mm, expect, interrupt_at=None):
    """Feed cmd_verify a board pinned at `board_base` in base_link."""
    rng = np.random.default_rng(0)

    # Invert the pipeline: verify computes T_base_flange @ X @ T_cam_target,
    # so hand it the T_cam_target that puts the board exactly where we want.
    with open('handeye_result.json') as f:
        X = np.array(json.load(f)['T_flange_cam'])

    state = {'i': 0}

    class FakeTarget:
        kind = 'charuco'

        def describe(self):
            return 'stub target'

        def detect(self, gray, K=None, dist=None):
            # Raise from exactly where the real one did: the signal landed
            # inside cv2.aruco.detectMarkers, so KeyboardInterrupt surfaced
            # here rather than as rclpy.ok() going false.
            if interrupt_at is not None and state['i'] >= interrupt_at:
                raise KeyboardInterrupt
            return np.zeros((8, 1, 2), np.float32), np.arange(8).reshape(-1, 1)

        def draw(self, view, corners, ids):
            pass

        def pose(self, corners, ids, K, dist):
            i = state['i']
            state['i'] += 1
            T_bf = H.tool_to_T(np.array(tool_poses[min(i, len(tool_poses) - 1)]))
            want = np.eye(4)
            want[:3, 3] = board_base + rng.normal(0, jitter_mm / 1000.0, 3)
            # T_cam_target such that T_bf @ X @ T_cam_target == want
            return np.linalg.inv(T_bf @ X) @ want

    class FakeNode:
        bgr = np.zeros((64, 64, 3), np.uint8)
        K = np.eye(3)
        dist = np.zeros(5)

        def destroy_node(self):
            pass

    n_iter = {'k': 0}

    fake_rclpy = types.SimpleNamespace(
        ok=lambda: n_iter['k'] < len(tool_poses),
        spin_once=lambda node, timeout_sec=0.0: n_iter.__setitem__('k', n_iter['k'] + 1),
        shutdown=lambda: None,
    )

    H.make_camera_node = lambda: (fake_rclpy, FakeNode())
    H.wait_for_camera = lambda rclpy, node, timeout=20.0: True
    H.make_target = lambda a, meta=None: FakeTarget()
    H.RobotFeed = lambda ip: FakeFeed(tool_poses)
    H.cv2.imshow = lambda *a, **k: None
    H.cv2.waitKey = lambda *a, **k: 0
    H.cv2.destroyAllWindows = lambda *a, **k: None
    H.cv2.putText = lambda *a, **k: None
    H.cv2.cvtColor = lambda img, code: img

    a = types.SimpleNamespace(
        result='handeye_result.json', ip='0.0.0.0',
        readings=f'/tmp/verify_readings_{name}.json',
        target=None, squares=None, square_mm=None, marker_mm=None,
        dict=None, max_rms=1.0)

    print(f'\n{"=" * 62}\n{name}\n{"=" * 62}')
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        H.cmd_verify(a)      # must return normally even when interrupted
    out = buf.getvalue()
    tail = '\n'.join(l for l in out.split('\n')
                     if l and not l.startswith('[verify]'))
    print(tail)

    ok = expect in out
    print(f'--> expected {expect}: {"OK" if ok else "*** MISMATCH ***"}')
    return ok


def main():
    # A real jog: 300 mm of travel and a big wrist turn.
    moving = [[-300 + 12 * i, 80 + 8 * i, 300 - 5 * i, 90 + 2 * i, 0, 96]
              for i in range(25)]
    # Arm essentially parked.
    parked = [[-300 + 0.5 * i, 80, 300, 90, 0, 96] for i in range(25)]

    board = np.array([-0.32, 0.013, 0.015])

    results = [
        run('good: steady board, arm jogged 300 mm', moving, board, 1.0, 'PASS'),
        run('swimming: board drifts 40 mm', moving, board, 40.0, 'FAIL'),
        run('parked: steady board, arm barely moved', parked, board, 1.0,
            'INCONCLUSIVE'),
        # The regression: an interrupt mid-loop must still produce the summary.
        run('interrupted mid-jog: summary must survive', moving, board, 1.0,
            'PASS', interrupt_at=18),
    ]
    print(f'\n{sum(results)}/{len(results)} scenarios behaved as expected')
    sys.exit(0 if all(results) else 1)


if __name__ == '__main__':
    main()
