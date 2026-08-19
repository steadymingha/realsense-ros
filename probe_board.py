#!/usr/bin/env python3
"""Look at one live camera frame and report what calibration target is in it.

Read-only: subscribes to the colour topic, grabs a frame, tries every common
ArUco dictionary plus a plain-checkerboard detector, writes an annotated PNG,
and prints what it found. Sends nothing anywhere.

    ./run_handeye.sh probe_board.py
"""
import sys
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, CameraInfo

COLOR_TOPIC = '/camera/d405/color/image_raw'
INFO_TOPIC = '/camera/d405/color/camera_info'

DICT_NAMES = [
    'DICT_4X4_50', 'DICT_4X4_100', 'DICT_4X4_250', 'DICT_4X4_1000',
    'DICT_5X5_50', 'DICT_5X5_100', 'DICT_5X5_250', 'DICT_5X5_1000',
    'DICT_6X6_50', 'DICT_6X6_100', 'DICT_6X6_250', 'DICT_6X6_1000',
    'DICT_7X7_50', 'DICT_7X7_100', 'DICT_7X7_250', 'DICT_7X7_1000',
    'DICT_ARUCO_ORIGINAL',
    'DICT_APRILTAG_16h5', 'DICT_APRILTAG_25h9',
    'DICT_APRILTAG_36h10', 'DICT_APRILTAG_36h11',
]


def get_dict(name):
    """cv2 4.5.4 uses Dictionary_get; 4.7+ uses getPredefinedDictionary."""
    const = getattr(cv2.aruco, name)
    try:
        return cv2.aruco.Dictionary_get(const)
    except AttributeError:
        return cv2.aruco.getPredefinedDictionary(const)


def get_params():
    try:
        return cv2.aruco.DetectorParameters_create()
    except AttributeError:
        return cv2.aruco.DetectorParameters()


class Grab(Node):
    def __init__(self):
        super().__init__('probe_board')
        self.bgr = None
        self.K = None
        self.create_subscription(Image, COLOR_TOPIC, self._img, 10)
        self.create_subscription(
            Image, COLOR_TOPIC, self._img,
            QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.create_subscription(CameraInfo, INFO_TOPIC, self._info, 10)

    def _img(self, msg):
        arr = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3)
        if msg.encoding == 'rgb8':
            arr = arr[:, :, ::-1]
        self.bgr = np.ascontiguousarray(arr)

    def _info(self, msg):
        self.K = np.array(msg.k).reshape(3, 3)


def main():
    rclpy.init()
    node = Grab()
    t0 = time.time()
    while rclpy.ok() and time.time() - t0 < 20:
        rclpy.spin_once(node, timeout_sec=0.1)
        if node.bgr is not None and node.K is not None:
            break
    if node.bgr is None:
        print('no image on', COLOR_TOPIC)
        sys.exit(1)

    img = node.bgr
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(f'frame {img.shape[1]}x{img.shape[0]}')
    if node.K is not None:
        print(f'K: fx={node.K[0,0]:.1f} fy={node.K[1,1]:.1f} '
              f'cx={node.K[0,2]:.1f} cy={node.K[1,2]:.1f}')

    print('\n--- ArUco dictionaries ---')
    params = get_params()
    best = None
    for name in DICT_NAMES:
        try:
            corners, ids, _ = cv2.aruco.detectMarkers(
                gray, get_dict(name), parameters=params)
        except cv2.error:
            continue
        n = 0 if ids is None else len(ids)
        if n:
            flat = sorted(int(i) for i in ids.flatten())
            print(f'  {name:22s} {n:3d} markers  ids={flat}')
            if best is None or n > best[1]:
                best = (name, n, corners, ids)
    if best is None:
        print('  none detected')

    print('\n--- plain checkerboard ---')
    found = []
    for cols in range(3, 12):
        for rows in range(3, 12):
            if cols <= rows:
                continue
            ok, _ = cv2.findChessboardCornersSB(gray, (cols, rows))
            if ok:
                found.append((cols, rows))
    print(f'  {found if found else "none"}')

    out = img.copy()
    if best is not None:
        cv2.aruco.drawDetectedMarkers(out, best[2], best[3])
        print(f'\nbest: {best[0]} with {best[1]} markers')
    cv2.imwrite('/tmp/handeye/board_probe.png', out)
    cv2.imwrite('/tmp/handeye/board_raw.png', img)
    print('wrote board_probe.png (annotated) and board_raw.png')

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
