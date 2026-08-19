#!/usr/bin/env python3
"""Eye-in-hand calibration for the D405 on the CR7 flange.

Solves T_flange_cam -- the fixed transform from the robot flange to the camera
optical frame -- so that a detected point becomes a robot target:

    P_base = T_base_flange @ T_flange_cam @ P_cam

Images come from realsense2_camera over the same topics aruco_test_gemini.py
uses, so the calibration shares the pipeline it is calibrating. The flange pose
comes straight off the controller's real-time feedback socket (port 30004),
which accepts multiple concurrent read-only clients -- so this runs alongside
bringup, or with no bringup at all. **Nothing is ever sent to the robot**: you
jog it by hand and this only watches.

T_base_flange is built from tool_vector_actual, whose rotation is intrinsic ZYX
-- Rz(rz) @ Ry(ry) @ Rx(rx) -- measured 2026-08-07 against pinocchio FK at a
live pose (0.46 deg residual; the next-best convention was 14 deg out, so this
is not a coin flip).

    !! tool_vector_actual is reported in the ACTIVE user/tool frame. Make sure
    !! the pendant has user=0 and tool=0 selected, or the "flange" this solves
    !! against is really some tool frame and everything downstream shifts.
    !! q_actual is recorded alongside every sample so the flange pose can be
    !! recomputed from FK later if that turns out to be wrong.

Usage (via ./run_handeye.sh, which runs these inside the ros2_dobot container)
----------------------------------------------------------------------------
    # camera up first, in its own terminal:
    ros2 launch realsense2_camera rs_launch.py camera_name:=d405 align_depth.enable:=true

    ./run_handeye.sh gen-board --squares 9x6 --square-mm 20   # printable board
    ./run_handeye.sh collect  --squares 9x6 --square-mm 20    # SPACE captures
    ./run_handeye.sh solve                                    # -> handeye_result.json
    ./run_handeye.sh verify                                   # jog; numbers must not move

`gen-board` and `solve` are pure offline maths and import no ROS, so they also
run under any plain python3 with cv2 + numpy.

Collecting well matters more than the solver
--------------------------------------------
Hand-eye is degenerate under pure translation: without rotation diversity the
solution is unconstrained along the camera axis and you get a confident,
completely wrong answer. So:

  * Rotate the wrist a lot between poses -- at least +-30 deg about two
    different axes, not just a slide sideways.
  * Keep the board 20-40 cm away and filling a good part of the frame.
  * Board flat on something rigid, matte print, and MEASURE the printed square
    with calipers instead of trusting the printer.
  * Vary distance too, not only angle.

`solve` reports the rotation spread of your poses and complains if it is too
small to trust.
"""
import argparse
import json
import math
import os
import socket
import struct
import sys
import threading
import time

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# robot real-time feedback (read-only, stdlib only -- no ROS needed for this)
# ---------------------------------------------------------------------------
ROBOT_IP = '192.168.5.1'
RT_PORT = 30004
RT_LEN = 1440

# Offsets into RealTimeData_t (dobot_bringup_v4/include/dobot_bringup/command.h)
OFF_LEN = 0
OFF_ROBOT_MODE = 24
OFF_Q_ACTUAL = 432          # double[6], degrees, CONTROLLER sign convention
OFF_QD_ACTUAL = 480         # double[6], deg/s
OFF_TOOL_VECTOR = 624       # double[6], x/y/z mm + rx/ry/rz deg

STILL_DPS = 0.5             # max joint speed that still counts as "stopped"

# Camera topics: the realsense2_camera names (camera_name:=d405), which the sim
# gazebo plugin also publishes -- same seam aruco_test_gemini.py sits on.
COLOR_TOPIC = '/camera/d405/color/image_raw'
INFO_TOPIC = '/camera/d405/color/camera_info'

SAMPLES_FILE = 'handeye_samples.json'
RESULT_FILE = 'handeye_result.json'


class RobotFeed(threading.Thread):
    """Keeps only the LATEST real-time frame.

    A daemon thread rather than an inline read per GUI iteration: the feed runs
    far faster than the display loop, so an inline reader falls further and
    further behind and ends up pairing images with poses from seconds ago --
    silently, since the numbers still look plausible.
    """

    def __init__(self, ip=ROBOT_IP):
        super().__init__(daemon=True)
        self.ip = ip
        self._lock = threading.Lock()
        self._frame = None
        self._err = None
        self._running = True

    def run(self):
        try:
            sock = socket.create_connection((self.ip, RT_PORT), timeout=3.0)
            sock.settimeout(3.0)
        except OSError as e:
            self._err = f'connect failed: {e}'
            return
        buf = b''
        while self._running:
            try:
                chunk = sock.recv(RT_LEN - len(buf))
            except socket.timeout:
                self._err = 'feed timed out'
                continue
            except OSError as e:
                self._err = f'recv failed: {e}'
                return
            if not chunk:
                self._err = 'feed closed'
                return
            buf += chunk
            if len(buf) < RT_LEN:
                continue
            packet, buf = buf[:RT_LEN], b''
            (length,) = struct.unpack_from('<H', packet, OFF_LEN)
            if length != RT_LEN:
                continue                     # out of sync; next read realigns
            frame = dict(
                mode=struct.unpack_from('<Q', packet, OFF_ROBOT_MODE)[0],
                q=np.array(struct.unpack_from('<6d', packet, OFF_Q_ACTUAL)),
                qd=np.array(struct.unpack_from('<6d', packet, OFF_QD_ACTUAL)),
                tool=np.array(struct.unpack_from('<6d', packet, OFF_TOOL_VECTOR)),
                wall=time.time(),
            )
            with self._lock:
                self._frame = frame
                self._err = None

    def latest(self, max_age_s=0.5):
        """Freshest frame, or None if stale/absent. Stale is treated as absent:
        a frozen feed must not be mistaken for a stationary arm."""
        with self._lock:
            f = self._frame
        if f is None or time.time() - f['wall'] > max_age_s:
            return None
        return f

    @property
    def error(self):
        return self._err

    def stop(self):
        self._running = False


# ---------------------------------------------------------------------------
# rotations
# ---------------------------------------------------------------------------

def Rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def Ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def Rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def tool_to_T(tool):
    """tool_vector_actual (mm, deg) -> 4x4 T_base_flange in metres.

    Rotation is intrinsic ZYX: Rz(rz) @ Ry(ry) @ Rx(rx). Verified against
    pinocchio FK, see module docstring.
    """
    rx, ry, rz = np.deg2rad(np.asarray(tool)[3:])
    T = np.eye(4)
    T[:3, :3] = Rz(rz) @ Ry(ry) @ Rx(rx)
    T[:3, 3] = np.asarray(tool)[:3] / 1000.0
    return T


def make_T(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t).reshape(3)
    return T


def rot_angle(R):
    """Geodesic magnitude of a rotation, degrees."""
    t = (np.trace(R) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, t))))


# ---------------------------------------------------------------------------
# board
# ---------------------------------------------------------------------------

def parse_squares(spec):
    """'9x6' -> (9, 6) inner corners (columns, rows)."""
    cols, rows = spec.lower().split('x')
    return int(cols), int(rows)


def object_points(squares, square_m):
    cols, rows = squares
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    return objp * square_m


def find_corners(gray, squares):
    """Sub-pixel inner corners, or None.

    findChessboardCornersSB is the newer detector: more robust to blur and
    uneven lighting, and already sub-pixel, so no cornerSubPix pass.
    """
    ok, corners = cv2.findChessboardCornersSB(
        gray, squares, flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
    if ok:
        return corners
    ok, corners = cv2.findChessboardCorners(
        gray, squares,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not ok:
        return None
    return cv2.cornerSubPix(
        gray, corners, (11, 11), (-1, -1),
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))


def board_pose(corners, squares, square_m, K, dist):
    """T_cam_target from one view, or None. IPPE is the planar-target solver."""
    objp = object_points(squares, square_m)
    try:
        ok, rvec, tvec = cv2.solvePnP(objp, corners, K, dist,
                                      flags=cv2.SOLVEPNP_IPPE)
    except cv2.error:
        ok = False
    if not ok:
        ok, rvec, tvec = cv2.solvePnP(objp, corners, K, dist,
                                      flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    return make_T(R, tvec)


def cmd_gen_board(a):
    """Write a printable checkerboard at exact scale.

    Printers lie. Print at 100% / "actual size", then measure a square with
    calipers and pass the MEASURED value to collect/solve -- the square size
    scales the whole translation part of the calibration linearly, so a 5%
    printing error is a 5% error in every distance you compute later.
    """
    cols, rows = parse_squares(a.squares)
    px = int(round(a.square_mm / 25.4 * a.dpi))
    # +1 because NxM inner corners needs (N+1)x(M+1) squares.
    img = np.zeros(((rows + 1) * px, (cols + 1) * px), np.uint8)
    for r in range(rows + 1):
        for c in range(cols + 1):
            if (r + c) % 2 == 0:
                img[r * px:(r + 1) * px, c * px:(c + 1) * px] = 255
    img = cv2.copyMakeBorder(img, px, px, px, px, cv2.BORDER_CONSTANT, value=127)
    cv2.imwrite(a.out, img)
    w_mm = (cols + 1) * a.square_mm
    h_mm = (rows + 1) * a.square_mm
    print(f'wrote {a.out}')
    print(f'  {cols}x{rows} inner corners, {a.square_mm} mm squares @ {a.dpi} dpi')
    print(f'  print area {w_mm:.0f} x {h_mm:.0f} mm '
          f'(grey border is a margin, not part of the board)')
    print('  print at 100% scale, then MEASURE a square and use that number')


# ---------------------------------------------------------------------------
# calibration target -- ChArUco or plain checkerboard behind one interface
# ---------------------------------------------------------------------------
# The board in the cell, read off its own printed legend (2026-08-07):
#     www.calib.io | 8x11 | Checker Size: 15 mm | Marker Size: 11 mm | DICT_4X4
# calib.io writes the layout rows-first, so 8x11 is 11 squares across by 8 down.
# `collect` tries both orientations at startup and reports which one detects --
# getting it backwards yields a pose that looks fine and is wrong.
CHARUCO_SQUARES = (11, 8)      # (squaresX, squaresY) = (across, down)
CHARUCO_SQUARE_MM = 15.0
CHARUCO_MARKER_MM = 11.0
CHARUCO_DICT = 'DICT_4X4_50'


def _aruco_dict(name):
    """cv2 4.5.4 (container) uses Dictionary_get; 4.7+ getPredefinedDictionary."""
    const = getattr(cv2.aruco, name)
    try:
        return cv2.aruco.Dictionary_get(const)
    except AttributeError:
        return cv2.aruco.getPredefinedDictionary(const)


def _aruco_params():
    try:
        return cv2.aruco.DetectorParameters_create()
    except AttributeError:
        return cv2.aruco.DetectorParameters()


def _charuco_board(squares, square_m, marker_m, dictionary):
    try:
        return cv2.aruco.CharucoBoard_create(
            squares[0], squares[1], square_m, marker_m, dictionary)
    except AttributeError:
        return cv2.aruco.CharucoBoard(squares, square_m, marker_m, dictionary)


class ChessTarget:
    """Plain checkerboard: all inner corners or nothing."""

    kind = 'chess'

    def __init__(self, squares, square_m):
        self.squares = squares
        self.square_m = square_m

    def detect(self, gray, K=None, dist=None):
        corners = find_corners(gray, self.squares)
        return (None, None) if corners is None else (corners, None)

    def pose(self, corners, ids, K, dist):
        return board_pose(corners, self.squares, self.square_m, K, dist)

    def pose_quality(self, corners, ids, K, dist):
        """(T, n_points, rms_px). A checkerboard is all-or-nothing, so every
        detected corner is an inlier by construction."""
        T = self.pose(corners, ids, K, dist)
        if T is None:
            return None, 0, None
        objp = object_points(self.squares, self.square_m)
        rvec, _ = cv2.Rodrigues(T[:3, :3])
        proj, _ = cv2.projectPoints(objp, rvec, T[:3, 3], K, dist)
        imgp = corners.reshape(-1, 2)
        rms = float(np.sqrt(((proj.reshape(-1, 2) - imgp) ** 2)
                            .sum(axis=1).mean()))
        return T, len(imgp), rms

    def draw(self, view, corners, ids):
        cv2.drawChessboardCorners(view, self.squares, corners, True)

    def describe(self):
        return (f'checkerboard {self.squares[0]}x{self.squares[1]} inner '
                f'corners, {self.square_m * 1000:.2f} mm squares')


class CharucoTarget:
    """ChArUco: the markers identify the corners, so partial views still solve.

    That is the whole reason to prefer it here -- at 20-40 cm a 165x120 mm board
    does not always fit the frame, and a plain checkerboard detects nothing at
    all when a single row falls outside.
    """

    kind = 'charuco'

    # Below this many interpolated corners the pose is too weak to trust.
    MIN_CORNERS = 8

    def __init__(self, squares, square_m, marker_m, dict_name):
        self.squares = squares
        self.square_m = square_m
        self.marker_m = marker_m
        self.dict_name = dict_name
        self.dictionary = _aruco_dict(dict_name)
        self.params = _aruco_params()
        self.board = _charuco_board(squares, square_m, marker_m, self.dictionary)

    # A charuco corner further than this from the fitted board is taken to be
    # mis-identified rather than merely noisy.
    RANSAC_PX = 2.0

    def detect(self, gray, K=None, dist=None):
        """K/dist are accepted for interface symmetry and deliberately NOT
        passed to interpolateCornersCharuco.

        cv2 4.5.4 uses them to run its own pose-based corner filter, and on this
        board that filter is destructive: measured on one frame, the same 66
        corners went from 56 RANSAC inliers at 0.18 px to 5 inliers and no
        usable pose. Interpolate purely from the marker layout and let
        pose_quality() do the rejecting.
        """
        corners, ids, _ = cv2.aruco.detectMarkers(
            gray, self.dictionary, parameters=self.params)
        if ids is None or len(ids) < 4:
            return None, None
        n, ch_corners, ch_ids = cv2.aruco.interpolateCornersCharuco(
            corners, ids, gray, self.board)
        if n < self.MIN_CORNERS:
            return None, None
        return ch_corners, ch_ids

    def pose(self, corners, ids, K, dist):
        T, _, _ = self.pose_quality(corners, ids, K, dist)
        return T

    def pose_quality(self, corners, ids, K, dist):
        """(T_cam_target, n_inliers, rms_px) -- RANSAC, not least squares.

        A glare patch or a single mis-read marker gives a handful of corners the
        wrong ID. Plain least squares does not reject them: measured on this
        board, 8 bad corners out of 66 dragged the pose to a 24 px reprojection
        error while a robust fit put 58 of them inside 3 px. That bad pose would
        have entered the sample set looking perfectly healthy.
        """
        objp = np.array(self.board.chessboardCorners,
                        np.float32)[ids.flatten()].reshape(-1, 3)
        imgp = corners.reshape(-1, 2).astype(np.float32)
        if len(objp) < 6:
            return None, 0, None
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            objp, imgp, K, dist, reprojectionError=self.RANSAC_PX,
            iterationsCount=200, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok or inliers is None or len(inliers) < self.MIN_CORNERS:
            return None, 0 if inliers is None else len(inliers), None
        idx = inliers.flatten()
        # Refit on the inliers alone: RANSAC's own estimate comes from a minimal
        # sample, so it is unbiased but noisy.
        ok, rvec, tvec = cv2.solvePnP(objp[idx], imgp[idx], K, dist, rvec, tvec,
                                      useExtrinsicGuess=True,
                                      flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return None, len(idx), None
        proj, _ = cv2.projectPoints(objp[idx], rvec, tvec, K, dist)
        rms = float(np.sqrt(((proj.reshape(-1, 2) - imgp[idx]) ** 2)
                            .sum(axis=1).mean()))
        R, _ = cv2.Rodrigues(rvec)
        return make_T(R, tvec), len(idx), rms

    def draw(self, view, corners, ids):
        cv2.aruco.drawDetectedCornersCharuco(view, corners, ids, (0, 255, 255))

    def describe(self):
        return (f'ChArUco {self.squares[0]}x{self.squares[1]} squares, '
                f'{self.square_m * 1000:.2f} mm checker / '
                f'{self.marker_m * 1000:.2f} mm marker, {self.dict_name}')


def make_target(a, meta=None):
    """Build the target from CLI args, falling back to what collect recorded."""
    meta = meta or {}
    kind = getattr(a, 'target', None) or meta.get('target', 'charuco')
    squares = (parse_squares(a.squares) if getattr(a, 'squares', None)
               else tuple(meta.get('squares',
                                   (9, 6) if kind == 'chess' else CHARUCO_SQUARES)))
    default_mm = 20.0 if kind == 'chess' else CHARUCO_SQUARE_MM
    mm = (getattr(a, 'square_mm', None) or meta.get('square_mm', default_mm))
    if kind == 'chess':
        return ChessTarget(squares, mm / 1000.0)
    marker_mm = (getattr(a, 'marker_mm', None)
                 or meta.get('marker_mm', CHARUCO_MARKER_MM))
    dict_name = getattr(a, 'dict', None) or meta.get('dict', CHARUCO_DICT)
    return CharucoTarget(squares, mm / 1000.0, marker_mm / 1000.0, dict_name)


def target_meta(target):
    """What solve/verify need to rebuild the same target later."""
    m = {'target': target.kind, 'squares': list(target.squares),
         'square_mm': target.square_m * 1000}
    if target.kind == 'charuco':
        m['marker_mm'] = target.marker_m * 1000
        m['dict'] = target.dict_name
    return m


# ---------------------------------------------------------------------------
# ROS image source
# ---------------------------------------------------------------------------

def make_camera_node():
    """Node holding the latest colour frame + intrinsics. ROS imported lazily
    so gen-board/solve stay usable outside a sourced ROS environment."""
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import Image, CameraInfo

    class CameraNode(Node):
        def __init__(self):
            super().__init__('handeye_calib')
            self.bgr = None
            self.bgr_wall = 0.0
            self.K = None
            self.dist = None
            # Two colour subscriptions, RELIABLE + BEST_EFFORT. A reliable
            # reader on the big colour stream can wedge and quietly serve stale
            # frames while the publisher is fine (measured in tag_vision_node,
            # 2026-07-15); a best-effort reader has no retransmit state and
            # cannot. Both feed the same callback, latest frame wins.
            self.create_subscription(Image, COLOR_TOPIC, self._img_cb, 10)
            self.create_subscription(
                Image, COLOR_TOPIC, self._img_cb,
                QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
            self.create_subscription(CameraInfo, INFO_TOPIC, self._info_cb, 10)

        def _img_cb(self, msg):
            arr = np.frombuffer(msg.data, np.uint8).reshape(
                msg.height, msg.width, 3)
            if msg.encoding == 'rgb8':
                arr = arr[:, :, ::-1]
            self.bgr = np.ascontiguousarray(arr)
            self.bgr_wall = time.time()

        def _info_cb(self, msg):
            if self.K is None:
                self.K = np.array(msg.k, dtype=float).reshape(3, 3)
                self.dist = (np.array(msg.d, dtype=float) if len(msg.d)
                             else np.zeros(5))
                self.get_logger().info(
                    f'CameraInfo: fx={self.K[0, 0]:.1f} fy={self.K[1, 1]:.1f} '
                    f'cx={self.K[0, 2]:.1f} cy={self.K[1, 2]:.1f} '
                    f'dist={np.round(self.dist, 5).tolist()}')

    rclpy.init()
    return rclpy, CameraNode()


def ros_shutdown(rclpy):
    """Shut the ROS context down at most once, and never raise.

    Ctrl+C reaches rclpy's signal handler first and shuts the context down
    itself, so an unguarded rclpy.shutdown() in a finally block raises and
    takes the caller's summary output down with it.
    """
    try:
        if rclpy.ok():
            rclpy.shutdown()
    except Exception:
        pass


def wait_for_camera(rclpy, node, timeout=20.0):
    print(f'waiting for {COLOR_TOPIC} + {INFO_TOPIC} ...')
    t0 = time.time()
    while rclpy.ok() and time.time() - t0 < timeout:
        rclpy.spin_once(node, timeout_sec=0.1)
        if node.bgr is not None and node.K is not None:
            print('camera up')
            return True
    print('!! no camera. Is realsense2_camera running with camera_name:=d405?')
    return False


# ---------------------------------------------------------------------------
# collect
# ---------------------------------------------------------------------------

# Auto-capture: how long the arm must sit still, and how far a pose must be
# from every pose already taken before it counts as a new one. Without the
# novelty gate a parked arm would fill the set with 200 copies of one view,
# which looks like plenty of data and constrains nothing.
AUTO_SETTLE_S = 0.8
AUTO_MIN_TRANS_M = 0.03
AUTO_MIN_ROT_DEG = 8.0


def _pose_is_new(T, taken):
    for U in taken:
        if (np.linalg.norm(T[:3, 3] - U[:3, 3]) < AUTO_MIN_TRANS_M
                and rot_angle(U[:3, :3].T @ T[:3, :3]) < AUTO_MIN_ROT_DEG):
            return False
    return True


def cmd_collect(a):
    target = make_target(a)
    print(f'target: {target.describe()}')

    feed = RobotFeed(a.ip)
    feed.start()
    rclpy, node = make_camera_node()
    if not wait_for_camera(rclpy, node):
        node.destroy_node()
        ros_shutdown(rclpy)
        return

    # Orientation check: a ChArUco board solved with squaresX/squaresY swapped
    # still returns a confident pose, just a wrong one. Detect once with both
    # and say which way round this board actually is.
    # The two orientations always interpolate the SAME number of corners --
    # inner corners are (X-1)(Y-1), which is symmetric under the swap -- so the
    # count cannot tell them apart. Reprojection error can: the wrong layout
    # puts the corners in the wrong places and the fit blows up.
    if target.kind == 'charuco':
        gray0 = cv2.cvtColor(node.bgr, cv2.COLOR_BGR2GRAY)
        flipped = CharucoTarget((target.squares[1], target.squares[0]),
                                target.square_m, target.marker_m,
                                target.dict_name)
        scored = []
        for cand in (target, flipped):
            c, i = cand.detect(gray0, node.K, node.dist)
            if c is None:
                scored.append((cand, 0, None))
                continue
            _, n, rms = cand.pose_quality(c, i, node.K, node.dist)
            scored.append((cand, n, rms))
            print(f'orientation {cand.squares}: {n} inliers, '
                  f'rms {"n/a" if rms is None else f"{rms:.2f} px"}')
        good = [s for s in scored if s[2] is not None]
        if good:
            best = min(good, key=lambda s: s[2])
            if best[0] is not target:
                print(f'  using {best[0].squares} (lower reprojection error)')
            target = best[0]

    samples, meta = [], {}
    if os.path.exists(a.samples) and not a.fresh:
        with open(a.samples) as f:
            meta = json.load(f)
        samples = meta['samples']
        print(f'[collect] appending to {len(samples)} existing samples '
              f'(--fresh to start over)')

    taken = [tool_to_T(np.array(s['tool_vector'])) for s in samples]

    def save(K, dist):
        """Write after every capture, not only at exit: an auto session runs for
        minutes while someone jogs, and a crash at minute nine should not throw
        away the first eight. Also lets progress be watched from outside."""
        out = dict(target_meta(target))
        out.update({'K': np.asarray(K).tolist(),
                    'dist': np.asarray(dist).tolist(),
                    'samples': samples})
        tmp = a.samples + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(out, f, indent=2)
        os.replace(tmp, a.samples)      # atomic; a reader never sees a half file

    auto = a.auto
    print(f'\nmode: {"AUTO (jog and stop; it captures itself)" if auto else "manual"}')
    print('SPACE = capture   A = toggle auto   U = undo last   Q/ESC = done\n')

    still_since = None
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.01)
            if node.bgr is None:
                continue
            img = node.bgr
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            corners, ids = target.detect(gray, node.K, node.dist)
            frame = feed.latest()

            view = img.copy()
            T_cam_target, n_in, rms = None, 0, None
            if corners is not None:
                target.draw(view, corners, ids)
                T_cam_target, n_in, rms = target.pose_quality(
                    corners, ids, node.K, node.dist)

            still = frame is not None and float(np.max(np.abs(frame['qd']))) < STILL_DPS
            fresh = time.time() - node.bgr_wall < 1.0
            sharp = rms is not None and rms <= a.max_rms
            ready = T_cam_target is not None and still and fresh and sharp
            still_since = (still_since or time.time()) if still else None

            if T_cam_target is not None:
                status = (f'{n_in}/{len(corners)} in  rms {rms:.2f}px @ '
                          f'{np.linalg.norm(T_cam_target[:3, 3]) * 100:.1f} cm')
                if not sharp:
                    status += f'  |  RMS > {a.max_rms} px'
            else:
                status = 'target NOT found'
            if frame is None:
                status += f'  |  NO ROBOT FEED ({feed.error or "connecting"})'
            elif not still:
                status += '  |  ARM MOVING'
            if not fresh:
                status += '  |  IMAGE STALE'

            capture = False
            if ready and auto:
                settled = time.time() - still_since >= AUTO_SETTLE_S
                new = _pose_is_new(tool_to_T(frame['tool']), taken)
                if settled and new:
                    capture = True
                elif settled:
                    status += '  |  auto: pose already covered, move more'

            cv2.putText(view, f'[{len(samples)}]{" AUTO" if auto else ""} {status}',
                        (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0) if ready else (0, 165, 255), 2, cv2.LINE_AA)
            cv2.imshow('hand-eye collect', view)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            if key == ord('a'):
                auto = not auto
                print(f'[collect] auto {"ON" if auto else "OFF"}')
            if key == ord('u') and samples:
                samples.pop()
                taken.pop()
                print(f'[collect] undo -> {len(samples)} samples')
            if key == ord(' '):
                if not ready:
                    # Refusing here is the whole point: a blurred corner set or
                    # a mid-motion pose pairs an image with a flange pose it
                    # does not belong to, and that error is invisible later.
                    print(f'[collect] refused ({status})')
                    continue
                capture = True

            if capture:
                samples.append({
                    'q_actual_deg': frame['q'].tolist(),
                    'tool_vector': frame['tool'].tolist(),
                    'corners': corners.reshape(-1, 2).tolist(),
                    'ids': (None if ids is None
                            else [int(i) for i in ids.flatten()]),
                    'stamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                })
                taken.append(tool_to_T(frame['tool']))
                still_since = None      # one capture per settle, not a burst
                save(node.K, node.dist)
                rot = max([rot_angle(taken[0][:3, :3].T @ U[:3, :3])
                           for U in taken[1:]] or [0.0])
                print(f'[collect] captured {len(samples)}  '
                      f'{n_in}/{len(corners)} inliers  rms {rms:.2f} px  '
                      f'rotation spread {rot:.0f} deg', flush=True)
    finally:
        cv2.destroyAllWindows()
        feed.stop()
        K, dist = node.K, node.dist
        node.destroy_node()
        ros_shutdown(rclpy)

    save(K, dist)
    print(f'\nwrote {a.samples} ({len(samples)} samples)')
    if len(samples) < 10:
        print('!! fewer than 10 poses -- collect more before trusting a solve')


# ---------------------------------------------------------------------------
# solve
# ---------------------------------------------------------------------------

METHODS = [
    ('TSAI', cv2.CALIB_HAND_EYE_TSAI),
    ('PARK', cv2.CALIB_HAND_EYE_PARK),
    ('HORAUD', cv2.CALIB_HAND_EYE_HORAUD),
    ('ANDREFF', cv2.CALIB_HAND_EYE_ANDREFF),
    ('DANIILIDIS', cv2.CALIB_HAND_EYE_DANIILIDIS),
]


def target_spread(T_base_flange, T_cam_target, X):
    """Consistency of a candidate X, in physical units.

    The board never moves, so T_base_target = T_base_flange @ X @ T_cam_target
    must come out identical for every sample. How much it does not is the
    calibration error, and unlike an algebraic AX=XB residual it is in mm and
    degrees -- the same units as the thing you care about.
    """
    poses = [B @ X @ C for B, C in zip(T_base_flange, T_cam_target)]
    ts = np.array([P[:3, 3] for P in poses])
    centre = ts.mean(axis=0)
    pos_rms_mm = float(np.sqrt(((ts - centre) ** 2).sum(axis=1).mean()) * 1000)
    R0 = poses[0][:3, :3]
    ang = [rot_angle(R0.T @ P[:3, :3]) for P in poses]
    return pos_rms_mm, float(np.mean(ang)), centre


def rigid_group(T_base_flange, T_cam_target, tol_deg=2.0):
    """Indices of the largest subset consistent with ONE rigid setup.

    AX = XB means the flange motion A_ij and the camera motion B_ij are
    conjugate rotations, so their rotation ANGLES are equal whatever X is:

        A_ij = T_flange_i^-1 @ T_flange_j
        B_ij = T_cam_i @ T_cam_j^-1
        angle(A_ij) == angle(B_ij)

    X cancels, so a mismatch cannot be blamed on a bad calibration -- it means
    the board was nudged, the camera shifted on its mount, or that view's pose
    is wrong. Solving over a set that spans such an event produces a confident
    answer that fits nothing, so find the largest coherent subset and use it.
    """
    n = len(T_base_flange)
    ok = np.ones((n, n), dtype=bool)
    for i in range(n):
        for j in range(i + 1, n):
            A = np.linalg.inv(T_base_flange[i]) @ T_base_flange[j]
            B = T_cam_target[i] @ np.linalg.inv(T_cam_target[j])
            agree = abs(rot_angle(A[:3, :3]) - rot_angle(B[:3, :3])) < tol_deg
            ok[i, j] = ok[j, i] = agree
    best = []
    for seed in range(n):
        group = [seed]
        for c in range(n):
            if c != seed and all(ok[c, g] for g in group):
                group.append(c)
        if len(group) > len(best):
            best = group
    return sorted(best)


def cmd_solve(a):
    with open(a.samples) as f:
        data = json.load(f)
    target = make_target(a, data)
    K = np.array(data['K'])
    dist = np.array(data['dist'])
    samples = data['samples']
    if len(samples) < 3:
        sys.exit(f'need at least 3 samples, have {len(samples)}')
    print(f'{len(samples)} samples, {target.describe()}')

    T_base_flange, T_cam_target = [], []
    for s in samples:
        corners = np.array(s['corners'], np.float32).reshape(-1, 1, 2)
        ids = (None if s.get('ids') is None
               else np.array(s['ids'], np.int32).reshape(-1, 1))
        T = target.pose(corners, ids, K, dist)
        if T is None:
            print('[solve] a sample failed pose estimation; skipped')
            continue
        T_cam_target.append(T)
        T_base_flange.append(tool_to_T(np.array(s['tool_vector'])))

    # Rigidity gate, before anything else: the whole method assumes one fixed
    # board and one fixed camera mount for the entire session.
    if not a.no_rigid_filter and len(T_base_flange) >= 4:
        keep = rigid_group(T_base_flange, T_cam_target, a.rigid_tol)
        dropped = len(T_base_flange) - len(keep)
        if dropped:
            print(f'\n!! rigidity: only {len(keep)}/{len(T_base_flange)} samples '
                  f'belong to one rigid setup; dropping {dropped}.')
            print('   Something moved mid-session -- the board was nudged, or '
                  'the camera shifted on its mount. If it is the mount, this '
                  'calibration will not hold and the mount needs fixing first.')
            print(f'   kept: {keep}')
            T_base_flange = [T_base_flange[i] for i in keep]
            T_cam_target = [T_cam_target[i] for i in keep]
        else:
            print('rigidity: all samples consistent with one fixed setup')

    # Rotation diversity gate. Hand-eye is degenerate under pure translation:
    # with too little rotation between poses the solution is unconstrained
    # along the camera axis and every method returns a confident wrong answer.
    rel = [rot_angle(T_base_flange[0][:3, :3].T @ T[:3, :3])
           for T in T_base_flange[1:]]
    spread = max(rel) if rel else 0.0
    print(f'rotation spread across poses: {spread:.1f} deg '
          f'(want > 40; > 60 is comfortable)')
    if spread < 25:
        print('!! too little rotation -- the solve WILL be wrong. Re-collect '
              'with the wrist turned about two different axes.')

    R_bf = [T[:3, :3] for T in T_base_flange]
    t_bf = [T[:3, 3] for T in T_base_flange]
    R_ct = [T[:3, :3] for T in T_cam_target]
    t_ct = [T[:3, 3] for T in T_cam_target]

    print('\n--- method            board spread (lower = better) ---')
    results = []
    for name, flag in METHODS:
        try:
            R_x, t_x = cv2.calibrateHandEye(R_bf, t_bf, R_ct, t_ct, method=flag)
        except cv2.error as e:
            print(f'  {name:12s} failed: {e}')
            continue
        X = make_T(R_x, t_x)
        pos_mm, ang_deg, centre = target_spread(T_base_flange, T_cam_target, X)
        results.append((pos_mm, ang_deg, name, X, centre))
        print(f'  {name:12s} {pos_mm:7.2f} mm   {ang_deg:6.2f} deg')

    if not results:
        sys.exit('every method failed')
    results.sort(key=lambda r: r[0])
    pos_mm, ang_deg, name, X, centre = results[0]

    t = X[:3, 3]
    print(f'\nbest: {name}   board position spread {pos_mm:.2f} mm rms, '
          f'orientation {ang_deg:.2f} deg')
    print('\nT_flange_cam (camera optical frame in the flange frame):')
    print(f'  xyz (m)  = [{t[0]:+.5f}, {t[1]:+.5f}, {t[2]:+.5f}]')
    print(f'  |xyz|    = {np.linalg.norm(t) * 1000:.1f} mm from the flange')
    print('  R =')
    for row in X[:3, :3]:
        print('    [' + ', '.join(f'{v:+.6f}' for v in row) + ']')
    print(f'\nboard centre in base_link: '
          f'[{centre[0]:+.4f}, {centre[1]:+.4f}, {centre[2]:+.4f}] m')

    if pos_mm > 10:
        print('\n!! spread over 10 mm. Usual causes, in order: too little '
              'rotation between poses; the printed square is not the size you '
              'passed; a tool/user frame other than 0 active on the pendant.')

    with open(a.out, 'w') as f:
        json.dump({
            'T_flange_cam': X.tolist(),
            'method': name,
            'spread_pos_mm': pos_mm,
            'spread_rot_deg': ang_deg,
            'n_samples': len(T_base_flange),
            'rotation_spread_deg': spread,
            'K': K.tolist(),
            'dist': dist.tolist(),
            'solved_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            **target_meta(target),
        }, f, indent=2)
    print(f'\nwrote {a.out}')


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------

def cmd_verify(a):
    """Live board pose in base_link. Jog the arm; the numbers must stay put.

    This is the accuracy check that works without ground truth: a wrong
    T_flange_cam makes the board appear to swim as the arm moves, and the size
    of the swim is the size of the error you will get on real targets.
    """
    with open(a.result) as f:
        res = json.load(f)
    X = np.array(res['T_flange_cam'])
    target = make_target(a, res)
    print(f'target: {target.describe()}')

    feed = RobotFeed(a.ip)
    feed.start()
    rclpy, node = make_camera_node()
    if not wait_for_camera(rclpy, node):
        node.destroy_node()
        ros_shutdown(rclpy)
        return

    seen = []
    seen_flange = []            # arm pose per reading -- proves the jog happened
    last_print = 0.0
    print('\nJog the arm between readings. Q/ESC to stop.\n')
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.01)
            if node.bgr is None:
                continue
            img = node.bgr
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            corners, ids = target.detect(gray, node.K, node.dist)
            frame = feed.latest()

            view = img.copy()
            text = 'target NOT found'
            colour = (0, 165, 255)
            if corners is not None:
                target.draw(view, corners, ids)
                T_cam_target = target.pose(corners, ids, node.K, node.dist)
                if T_cam_target is not None and frame is not None:
                    p = (tool_to_T(frame['tool']) @ X @ T_cam_target)[:3, 3]
                    text = f'base_link  x={p[0]:+.4f}  y={p[1]:+.4f}  z={p[2]:+.4f}'
                    if float(np.max(np.abs(frame['qd']))) < STILL_DPS:
                        colour = (0, 255, 0)
                        seen.append(p)
                        seen_flange.append(tool_to_T(frame['tool']))
                        if len(seen) > 1:
                            sp = np.array(seen)
                            rng_mm = (sp.max(axis=0) - sp.min(axis=0)) * 1000
                            text += f'   spread {np.max(rng_mm):.1f} mm'
                            # Also to the console, throttled: the GUI is on the
                            # robot's own monitor, which is not always where the
                            # person watching happens to be.
                            now = time.time()
                            if now - last_print > 2.0:
                                last_print = now
                                print(f'[verify] n={len(seen):4d}  '
                                      f'base=({p[0]:+.4f}, {p[1]:+.4f}, '
                                      f'{p[2]:+.4f})  spread '
                                      f'{np.max(rng_mm):.1f} mm', flush=True)
                    else:
                        text += '   (moving)'
            cv2.putText(view, text, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        colour, 2, cv2.LINE_AA)
            cv2.imshow('hand-eye verify', view)
            if (cv2.waitKey(1) & 0xFF) in (ord('q'), 27):
                break
    except KeyboardInterrupt:
        # The expected way to stop: the operator has finished jogging. Ctrl+C
        # can surface either as rclpy.ok() going false or as this, depending on
        # where the signal lands, and the summary has to survive both.
        print('\n-- interrupted; summarising the readings collected so far')
    except Exception as e:
        # Includes rclpy's ExternalShutdownException. A partial summary beats
        # losing several thousand readings to a traceback.
        print(f'\n-- verify loop ended on {type(e).__name__}: {e}')
        print('-- summarising the readings collected so far')
    finally:
        cv2.destroyAllWindows()
        feed.stop()
        node.destroy_node()
        ros_shutdown(rclpy)

    if len(seen) > 1:
        # Robust statistics, not max-minus-min. A planar target admits a mirror
        # pose that fits the image almost as well, and when the solver takes it
        # the board lands hundreds of mm away. Measured here: 3 readings in 100
        # were 300-400 mm out while the other 97 sat inside 10 mm. A range
        # reports that as "2470 mm spread" and buries a good calibration.
        sp = np.array(seen)
        med = np.median(sp, axis=0)
        dev = np.linalg.norm(sp - med, axis=1) * 1000
        out = dev > 20.0
        print(f'\n{len(sp)} still readings')
        print(f'  median position : {np.round(med, 4).tolist()} m')
        print(f'  deviation mm    : median {np.median(dev):.2f}  '
              f'p90 {np.percentile(dev, 90):.2f}  p95 {np.percentile(dev, 95):.2f}')
        print(f'  outliers > 20mm : {out.sum()}/{len(dev)} '
              f'({100 * out.mean():.1f}%)  max {dev.max():.1f} mm')
        # How far the arm travelled between readings. Without this the
        # spread is ungradeable: a wrong X held still also looks perfect.
        if seen_flange:
            fp = np.array([T[:3, 3] for T in seen_flange])
            travel_mm = np.max(fp.max(axis=0) - fp.min(axis=0)) * 1000
            rots = [rot_angle(seen_flange[0][:3, :3].T @ T[:3, :3])
                    for T in seen_flange[1:]]
            travel_deg = max(rots) if rots else 0.0
            print(f'  arm travel      : {travel_mm:.0f} mm, '
                  f'{travel_deg:.0f} deg of wrist rotation')
        else:
            travel_mm = travel_deg = 0.0

        good = sp[~out]
        if len(good) > 1:
            print(f'  per-axis std mm : '
                  f'{np.round(good.std(axis=0) * 1000, 2).tolist()}')
        if out.sum() and out.mean() < 0.2:
            print('  a few outliers among many good readings is the planar '
                  'mirror pose, not calibration error -- disambiguate with '
                  'the depth normal to remove them.')
        elif out.sum():
            print(f'  {100 * out.mean():.0f}% of readings are outliers. That '
                  'is too many to be the planar mirror pose; the board is '
                  'genuinely moving with the arm.')

        # Keep the raw readings. Re-analysing the last run meant scraping a
        # throttled console log because nothing on disk survived it.
        with open(a.readings, 'w') as f:
            json.dump({'xyz_base': sp.tolist(),
                       'flange': [T.tolist() for T in seen_flange],
                       'result': a.result}, f)
        print(f'  readings written to {a.readings}')

        # An explicit verdict. The thresholds are what the pick-place sequence
        # needs: it hovers 30-50 mm above the target and descends on contact,
        # so a board that swims more than ~10 mm will miss the hover window.
        inlier_dev = dev[~out]
        p95 = np.percentile(inlier_dev, 95) if len(inlier_dev) else float('inf')
        print()
        if travel_mm < 100 or travel_deg < 30:
            print(f'INCONCLUSIVE: the arm only moved {travel_mm:.0f} mm / '
                  f'{travel_deg:.0f} deg. Jog it over at least 100 mm and 30 '
                  'deg of wrist rotation -- a stationary arm cannot disprove '
                  'a bad calibration.')
        elif p95 < 10.0:
            print(f'PASS: board holds to {p95:.1f} mm (p95 of inliers) over '
                  f'{travel_mm:.0f} mm / {travel_deg:.0f} deg of arm motion.')
        else:
            print(f'FAIL: board swims {p95:.1f} mm (p95 of inliers) over '
                  f'{travel_mm:.0f} mm / {travel_deg:.0f} deg of arm motion. '
                  'Do not use this T_flange_cam.')


# ---------------------------------------------------------------------------
# selftest -- no camera, no robot
# ---------------------------------------------------------------------------

def cmd_selftest(a):
    """Check the two halves separately: does the vision path work, and does the
    maths path work. Run it after any edit -- it needs no hardware."""
    ok = True

    print('=== 1. detect + PnP on a rendered board ===')
    squares = (9, 6)
    px = 60
    img = np.zeros(((squares[1] + 1) * px, (squares[0] + 1) * px), np.uint8)
    for r in range(squares[1] + 1):
        for c in range(squares[0] + 1):
            if (r + c) % 2 == 0:
                img[r * px:(r + 1) * px, c * px:(c + 1) * px] = 255
    img = cv2.copyMakeBorder(img, px, px, px, px, cv2.BORDER_CONSTANT, value=127)
    corners = find_corners(img, squares)
    n = 0 if corners is None else len(corners)
    print(f'  corners: {n} / {squares[0] * squares[1]}')
    ok &= n == squares[0] * squares[1]

    print('\n=== 2. synthetic hand-eye round trip ===')
    rng = np.random.default_rng(0)
    X_true = make_T(Rz(np.deg2rad(-90)) @ Rx(np.deg2rad(-90)), [0.090, 0.0, 0.040])
    T_base_target = make_T(Rz(np.deg2rad(15)) @ Rx(np.deg2rad(180)),
                           [-0.40, 0.10, 0.05])
    T_bf, T_ct = [], []
    for _ in range(15):
        R = (Rz(rng.uniform(-0.7, 0.7)) @ Ry(rng.uniform(-0.7, 0.7))
             @ Rx(rng.uniform(-0.7, 0.7)))
        B = make_T(R, np.array([-0.35, 0.10, 0.30]) + rng.uniform(-0.08, 0.08, 3))
        T_bf.append(B)
        T_ct.append(np.linalg.inv(X_true) @ np.linalg.inv(B) @ T_base_target)
    for name, flag in METHODS:
        R_x, t_x = cv2.calibrateHandEye([b[:3, :3] for b in T_bf],
                                        [b[:3, 3] for b in T_bf],
                                        [c[:3, :3] for c in T_ct],
                                        [c[:3, 3] for c in T_ct], method=flag)
        X = make_T(R_x, t_x)
        d_mm = np.linalg.norm(X[:3, 3] - X_true[:3, 3]) * 1000
        d_deg = rot_angle(X_true[:3, :3].T @ X[:3, :3])
        spread, _, centre = target_spread(T_bf, T_ct, X)
        print(f'  {name:12s} err {d_mm:7.4f} mm {d_deg:7.4f} deg   '
              f'spread {spread:7.4f} mm')
        ok &= d_mm < 0.1 and d_deg < 0.1

    print('\n=== 3. tool_to_T against a live frame recorded 2026-08-07 ===')
    T = tool_to_T(np.array([-357.649, 105.967, 297.336, 91.908, -9.589, 93.659]))
    want = np.array([-0.35765, 0.10597, 0.29734])
    err_mm = np.linalg.norm(T[:3, 3] - want) * 1000
    det = float(np.linalg.det(T[:3, :3]))
    print(f'  translation err {err_mm:.4f} mm   det(R) {det:.9f}')
    ok &= err_mm < 0.02 and abs(det - 1.0) < 1e-9

    print('\nPASS' if ok else '\nFAIL')
    sys.exit(0 if ok else 1)


# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)

    def common(sp, square_default=None):
        sp.add_argument('--ip', default=ROBOT_IP)
        sp.add_argument('--samples', default=SAMPLES_FILE)
        sp.add_argument('--target', choices=('charuco', 'chess'), default=None,
                        help='default charuco; recorded at collect time')
        sp.add_argument('--squares', default=None,
                        help='charuco: squares across x down (e.g. 11x8); '
                             'chess: inner corners (e.g. 9x6)')
        sp.add_argument('--square-mm', type=float, default=square_default,
                        help='measured checker size in mm; on solve/verify it '
                             'overrides the value recorded at collect time')
        sp.add_argument('--marker-mm', type=float, default=None,
                        help='charuco marker size in mm')
        sp.add_argument('--dict', default=None, help='charuco ArUco dictionary')
        sp.add_argument('--max-rms', type=float, default=1.0,
                        help='reject a view whose inlier reprojection RMS '
                             'exceeds this many pixels (default 1.0)')

    g = sub.add_parser('gen-board', help='write a printable checkerboard')
    g.add_argument('--squares', default='9x6', help='inner corners, e.g. 9x6')
    g.add_argument('--square-mm', type=float, default=20.0)
    g.add_argument('--dpi', type=int, default=300)
    g.add_argument('--out', default='board.png')
    g.set_defaults(func=cmd_gen_board)

    c = sub.add_parser('collect', help='capture board views + flange poses')
    c.add_argument('--fresh', action='store_true',
                   help='discard existing samples')
    c.add_argument('--auto', action='store_true',
                   help='capture by itself whenever the arm settles in a pose '
                        'not already covered -- no keypress while jogging')
    common(c)
    c.set_defaults(func=cmd_collect)

    s = sub.add_parser('solve', help='solve T_flange_cam')
    s.add_argument('--out', default=RESULT_FILE)
    s.add_argument('--rigid-tol', type=float, default=2.0,
                   help='degrees of AX=XB angle mismatch still counted as one '
                        'rigid setup (default 2.0)')
    s.add_argument('--no-rigid-filter', action='store_true',
                   help='keep every sample even if the setup moved mid-session')
    common(s)
    s.set_defaults(func=cmd_solve)

    v = sub.add_parser('verify', help='live check that the board sits still')
    v.add_argument('--result', default=RESULT_FILE)
    common(v)
    v.add_argument('--readings', default='verify_readings.json',
                   help='where to keep the raw per-reading data')
    v.set_defaults(func=cmd_verify)

    t = sub.add_parser('selftest', help='offline check; no camera or robot')
    t.set_defaults(func=cmd_selftest)

    a = p.parse_args()
    a.func(a)


if __name__ == '__main__':
    main()
