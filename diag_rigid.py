#!/usr/bin/env python3
"""Was the setup rigid? A test that needs no calibration at all.

AX = XB says: for any two samples i,j, the flange motion A_ij and the camera
motion B_ij are conjugate rotations, so their rotation ANGLES must be equal --
whatever X happens to be. X cancels.

    A_ij = T_flange_i^-1 @ T_flange_j
    B_ij = T_cam_i @ T_cam_j^-1
    angle(A_ij) == angle(B_ij)   for a rigid camera and a fixed board

So a large angle mismatch cannot be blamed on a bad calibration. It means the
board moved, the camera shifted on its mount, or that view's pose is wrong.
Grouping samples by mutual agreement then shows exactly when it happened.

Read-only.
"""
import json
import sys

import numpy as np
import cv2

sys.path.insert(0, '/tmp/handeye')
import handeye_calib as h


def main():
    with open('/tmp/handeye/handeye_samples.json') as f:
        d = json.load(f)
    K, dist = np.array(d['K']), np.array(d['dist'])
    a = type('A', (), dict(target=None, squares=None, square_mm=None,
                           marker_mm=None, dict=None))()
    target = h.make_target(a, d)

    Tb, Tc, idx = [], [], []
    for k, s in enumerate(d['samples']):
        c = np.array(s['corners'], np.float32).reshape(-1, 1, 2)
        i = np.array(s['ids'], np.int32).reshape(-1, 1)
        T, n, rms = target.pose_quality(c, i, K, dist)
        if T is None:
            continue
        Tb.append(h.tool_to_T(np.array(s['tool_vector'])))
        Tc.append(T)
        idx.append(k)
    n = len(Tb)
    print(f'{n} samples\n')

    M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            A = np.linalg.inv(Tb[i]) @ Tb[j]
            B = Tc[i] @ np.linalg.inv(Tc[j])
            M[i, j] = abs(h.rot_angle(A[:3, :3]) - h.rot_angle(B[:3, :3]))

    print('=== angle mismatch |angle(A_ij) - angle(B_ij)|, degrees ===')
    print('(near 0 = rigid pair; large = something moved between them)')
    off = M[~np.eye(n, dtype=bool)]
    print(f'  overall: median {np.median(off):.2f}  mean {off.mean():.2f}  '
          f'max {off.max():.2f}\n')

    # Greedy largest group whose members all agree with each other.
    TOL = 2.0
    adj = M < TOL
    best = []
    for seed in range(n):
        group = [seed]
        for c in range(n):
            if c == seed:
                continue
            if all(adj[c, g] for g in group):
                group.append(c)
        if len(group) > len(best):
            best = group
    best = sorted(best)
    print(f'=== largest mutually rigid group (tol {TOL} deg) ===')
    print(f'  {len(best)}/{n} samples: {[idx[i] for i in best]}\n')

    print('=== per-sample: how many others it agrees with ===')
    agree = (adj.sum(axis=1)).astype(int)
    for i in range(n):
        bar = '#' * int(agree[i] * 40 / n)
        print(f'  {idx[i]:3d}  {agree[i]:3d}/{n-1}  {bar}')


if __name__ == '__main__':
    main()
