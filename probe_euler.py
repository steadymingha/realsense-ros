"""Read-only probe: pin down the Euler convention of the controller's
tool_vector_actual by comparing it with pinocchio FK of the SAME live pose.

Sends nothing to the robot -- it only reads the real-time feed (port 30004,
multi-client) and runs FK locally. Run inside the ros2_dobot container.
"""
import math
import socket
import struct
import sys

import numpy as np
import pinocchio as pin

sys.path.insert(0, '/root/dobot_ws/src/DOBOT_6Axis_ROS2_V4')
from cr7_pnp.model import ReachabilityModel  # noqa: E402

JOINT_SIGN_REAL = np.array([-1.0, 1.0, 1.0, 1.0, -1.0, -1.0])
RT_PORT, RT_LEN = 30004, 1440
OFF_Q_ACTUAL, OFF_TOOL_VECTOR = 432, 624


def read_rt(ip):
    with socket.create_connection((ip, RT_PORT), timeout=3.0) as s:
        s.settimeout(3.0)
        buf = b''
        while len(buf) < RT_LEN:
            chunk = s.recv(RT_LEN - len(buf))
            if not chunk:
                raise ConnectionError('feed closed early')
            buf += chunk
    return (struct.unpack_from('<6d', buf, OFF_Q_ACTUAL),
            struct.unpack_from('<6d', buf, OFF_TOOL_VECTOR))


def Rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def Ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def Rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


AXIS = {'X': Rx, 'Y': Ry, 'Z': Rz}


def angle_between(Ra, Rb):
    """Geodesic angle between two rotations, degrees."""
    t = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, t))))


def main():
    ip = sys.argv[1] if len(sys.argv) > 1 else '192.168.5.1'
    q_ctrl, tool = read_rt(ip)
    print('q_actual   (deg, controller):', [round(v, 3) for v in q_ctrl])
    print('tool_vector(mm/deg)         :', [round(v, 3) for v in tool])

    q_urdf = np.deg2rad(np.array(q_ctrl) * JOINT_SIGN_REAL)

    m = ReachabilityModel()
    qp = m.pin_q(list(q_urdf))
    pin.forwardKinematics(m.model, m.data, qp)
    pin.updateFramePlacements(m.model, m.data)
    base = m.data.oMf[m.model.getFrameId('base_link')]
    flange = base.actInv(m.data.oMf[m.frame_id])

    p_fk = flange.translation
    p_rt = np.array(tool[:3]) / 1000.0
    print(f'\nflange pos FK   (m): {p_fk}')
    print(f'flange pos feed (m): {p_rt}')
    print(f'translation error  : {np.linalg.norm(p_fk - p_rt) * 1000:.2f} mm')

    rx, ry, rz = np.deg2rad(tool[3:])
    ang = {'X': rx, 'Y': ry, 'Z': rz}
    R_fk = flange.rotation

    print('\n--- Euler convention candidates (angle vs FK rotation) ---')
    rows = []
    for order in ('XYZ', 'XZY', 'YXZ', 'YZX', 'ZXY', 'ZYX'):
        a, b, c = order
        R_in = AXIS[a](ang[a]) @ AXIS[b](ang[b]) @ AXIS[c](ang[c])
        rows.append((angle_between(R_fk, R_in), f'intrinsic {order}  '
                     f'(R{a}@R{b}@R{c})'))
    for err, label in sorted(rows):
        print(f'  {err:8.3f} deg   {label}')


if __name__ == '__main__':
    main()
