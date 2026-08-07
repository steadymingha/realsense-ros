import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2, time
import numpy as np

class RobustDepthArucoTracker(Node):
    def __init__(self):
        super().__init__('aruco_tracker')
        self.bridge = CvBridge()
        
        self.camera_matrix = None
        self.dist_coeffs = None
        self.target_aruco_frame = None
        self.base_marker_size = 0.03    # ID 0 (원점) 검은 테두리 바깥 변 길이 (m)
        self.target_marker_size = 0.02  # ID 1 (타겟) — 위치는 depth로 구하므로 현재 PnP엔 미사용
        
        # 최신 프레임을 저장할 버퍼
        self.latest_color = None
        self.latest_depth = None
        self.mouse_clicked_point = None

        # 스페이스바 측정: 30프레임 수집 → 평균±표준편차
        self.sample_buf = None
        self.initial_target_pos = None


        try:
            self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_APRILTAG_36h11)
            self.aruco_params = cv2.aruco.DetectorParameters_create()
        except AttributeError:
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
            self.aruco_params = cv2.aruco.DetectorParameters()

        # Realsense 특성을 고려한 QoS (Sensor Data = Best Effort)
        custom_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)

        # 토픽 구독 (동기화 없이 각각 최신 상태 유지)
        self.info_sub = self.create_subscription(CameraInfo, '/camera/d405/color/camera_info', self.info_cb, 10)
        self.color_sub = self.create_subscription(Image, '/camera/d405/color/image_raw', self.color_cb, 10)
        self.depth_sub = self.create_subscription(Image, '/camera/d405/aligned_depth_to_color/image_raw', self.depth_cb, custom_qos)

        # 30Hz로 화면 업데이트 및 처리
        self.timer = self.create_timer(0.033, self.process_frame)

        cv2.namedWindow("Real Target Tracker")
        cv2.setMouseCallback("Real Target Tracker", self.mouse_cb)

    def info_cb(self, msg):
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k).reshape(3, 3)
            self.dist_coeffs = np.array(msg.d)
            self.get_logger().info("Camera Info load complete.")
            self.destroy_subscription(self.info_sub)

    def color_cb(self, msg):
        self.latest_color = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def depth_cb(self, msg):
        self.latest_depth = self.bridge.imgmsg_to_cv2(msg, "16UC1")

    def mouse_cb(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.mouse_clicked_point = (x, y)

    def marker_plane_normal(self, depth_img, quad, scale=2.0):
        """마커 주변까지 확장한 영역의 depth에 평면을 피팅해 법선(카메라 좌표계) 반환.

        scale: 마커 사각형을 중심 기준으로 확대하는 배율. 마커가 붙은 평평한 면까지
        같이 샘플링해 기울기 정확도를 높인다. 면을 벗어난 픽셀(배경 등)은
        잔차 5mm 초과 기각으로 자동 제거된다. 실패 시 None.
        """
        qc = quad.mean(axis=0)
        quad = qc + (quad - qc) * scale
        mask = np.zeros(depth_img.shape, np.uint8)
        cv2.fillConvexPoly(mask, quad.astype(np.int32), 1)
        ys, xs = np.nonzero((mask > 0) & (depth_img > 0))
        if len(xs) < 30:
            return None
        z = depth_img[ys, xs].astype(np.float64) * 0.001
        pts_px = np.stack([xs, ys], axis=1).astype(np.float32).reshape(-1, 1, 2)
        und = cv2.undistortPoints(pts_px, self.camera_matrix, self.dist_coeffs).reshape(-1, 2)
        pts = np.column_stack([und[:, 0] * z, und[:, 1] * z, z])

        n = None
        for _ in range(3):  # 피팅 → 평면 밖 점 기각 → 재피팅
            centroid = pts.mean(axis=0)
            _, _, vt = np.linalg.svd(pts - centroid, full_matrices=False)
            n = vt[2]
            inlier = np.abs((pts - centroid) @ n) < 0.005
            if inlier.all() or inlier.sum() < 30:
                break
            pts = pts[inlier]
        return -n if n[2] > 0 else n  # 카메라 쪽을 향하도록

    def get_robust_z(self, depth_img, x, y, roi=5):
        h, w = depth_img.shape
        half = roi // 2
        x_min, x_max = max(0, x - half), min(w, x + half + 1)
        y_min, y_max = max(0, y - half), min(h, y + half + 1)
        roi_d = depth_img[y_min:y_max, x_min:x_max]
        valid_d = roi_d[roi_d > 0]
        if len(valid_d) == 0: return 0.0
        return np.median(valid_d) * 0.001

    def finish_measurement(self):
        arr = np.array(self.sample_buf)
        self.sample_buf = None
        m, sd = arr.mean(axis=0), arr.std(axis=0)

        pc = self.current_P_cam
        br, bt = self.current_base_pose
        self.get_logger().info(
            f"[debug] P_cam X:{pc[0]:.1f} Y:{pc[1]:.1f} Z:{pc[2]:.1f}cm | "
            f"base_t X:{bt[0]*100:.1f} Y:{bt[1]*100:.1f} Z:{bt[2]*100:.1f}cm | "
            f"base_r {np.degrees(br[0]):.1f} {np.degrees(br[1]):.1f} {np.degrees(br[2]):.1f}deg"
        )
        self.get_logger().info(
            f"measured X:{m[0]:.2f}±{sd[0]:.2f} Y:{m[1]:.2f}±{sd[1]:.2f} Z:{m[2]:.2f}±{sd[2]:.2f} cm"
        )

        if self.initial_target_pos is None:
            self.initial_target_pos = m
            self.get_logger().info(f"origin locking X:{m[0]:.2f} Y:{m[1]:.2f} Z:{m[2]:.2f}cm")
        else:
            e = m - self.initial_target_pos
            self.get_logger().info(f"error X: {e[0]:+.2f}cm, Y: {e[1]:+.2f}cm, Z: {e[2]:+.2f}cm")

        # 정확도 산출용 CSV 축적 (analyze_accuracy.py로 통계 계산)
        import csv, os
        new_file = not os.path.exists("measurements.csv")
        with open("measurements.csv", "a", newline="") as fp:
            w = csv.writer(fp)
            if new_file:
                w.writerow(["time", "mean_x", "mean_y", "mean_z",
                            "std_x", "std_y", "std_z",
                            "pcam_x", "pcam_y", "pcam_z",
                            "base_tx", "base_ty", "base_tz"])
            w.writerow([f"{time.time():.1f}",
                        *[f"{v:.3f}" for v in m], *[f"{v:.3f}" for v in sd],
                        *[f"{v:.3f}" for v in pc], *[f"{v * 100:.3f}" for v in bt]])

    def process_frame(self):
        # 1. 컬러와 뎁스 이미지가 모두 들어오는지 확인 (다시 뎁스 필수!)
        if self.camera_matrix is None or self.latest_color is None or self.latest_depth is None:
            return

        img = self.latest_color.copy()
        depth_img = self.latest_depth.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(img, corners, ids)
            
            # base(ID 0) 마커의 3D 모서리 좌표 정의
            s = self.base_marker_size
            obj_points = np.array([
                [-s/2,  s/2, 0],
                [ s/2,  s/2, 0],
                [ s/2, -s/2, 0],
                [-s/2, -s/2, 0]
            ], dtype=np.float32)

            base_idx = np.where(ids == 0)[0]
            target_idx = np.where(ids == 1)[0]

            T_cam2base = None

            # ========================================================
            # [1단계] ID 0 (원점) 자세: PnP(면내 회전) + depth(기울기/위치) 융합
            # ========================================================
            if len(base_idx) > 0:
                bc = corners[base_idx[0]][0]
                # solvePnPGeneric은 평면 마커의 두 해(flip ambiguity)를 모두 반환
                n_sol, rvecs, tvecs, reproj_err = cv2.solvePnPGeneric(
                    obj_points, bc, self.camera_matrix, self.dist_coeffs,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE
                )
                if n_sol > 0:
                    n_depth = self.marker_plane_normal(depth_img, bc)

                    # flip 해소: depth 평면 법선과 마커 Z축(R[:,2])이 일치하는 해 선택
                    errs = np.asarray(reproj_err).flatten()
                    best = int(np.argmin(errs))
                    if n_sol > 1 and n_depth is not None:
                        dots = []
                        for r in rvecs[:n_sol]:
                            R_c, _ = cv2.Rodrigues(r)
                            dots.append(float(n_depth @ R_c[:, 2]))
                        best = int(np.argmax(dots))

                    R_0, _ = cv2.Rodrigues(rvecs[best])
                    t_0 = tvecs[best].flatten()

                    # [보정 1] 기울기: 마커 Z축을 depth 법선에 최소 회전으로 정렬
                    # (면내 회전은 PnP 것을 유지 — 작은 마커도 면내 회전은 정확함)
                    if n_depth is not None:
                        z_axis = R_0[:, 2]
                        c = float(z_axis @ n_depth)
                        if c > -0.999:
                            v = np.cross(z_axis, n_depth)
                            vx = np.array([[0, -v[2], v[1]],
                                           [v[2], 0, -v[0]],
                                           [-v[1], v[0], 0]])
                            R_0 = (np.eye(3) + vx + vx @ vx / (1.0 + c)) @ R_0

                    # [보정 2] 위치: 타겟과 동일하게 중심 픽셀 + depth로
                    # (PnP 스케일/마커 크기 오차 및 depth와의 z bias 불일치 제거)
                    bcx, bcy = float(bc[:, 0].mean()), float(bc[:, 1].mean())
                    bz = self.get_robust_z(depth_img, int(bcx), int(bcy))
                    if bz > 0.0:
                        und0 = cv2.undistortPoints(
                            np.array([[[bcx, bcy]]], dtype=np.float32),
                            self.camera_matrix, self.dist_coeffs
                        ).flatten()
                        t_0 = np.array([und0[0] * bz, und0[1] * bz, bz])

                    T_cam2base = np.eye(4)
                    T_cam2base[:3, :3] = R_0
                    T_cam2base[:3, 3] = t_0

                    rvec_draw, _ = cv2.Rodrigues(R_0)
                    # 디버그용: 스페이스바 로그에서 base 자세 확인
                    self.current_base_pose = (rvec_draw.flatten().copy(), t_0.copy())
                    cv2.drawFrameAxes(img, self.camera_matrix, self.dist_coeffs,
                                      rvec_draw, t_0.reshape(3, 1), 0.05)

# ========================================================
            # [2단계] 타겟(지우개) 3D 좌표 추출 (Depth 센서 기반)
            # ========================================================
            if T_cam2base is not None and len(target_idx) > 0:
                target_corners = corners[target_idx[0]][0]
                cxf = float(np.mean(target_corners[:, 0]))
                cyf = float(np.mean(target_corners[:, 1]))
                cx, cy = int(cxf), int(cyf)

                z_val = self.get_robust_z(depth_img, cx, cy)

                if z_val > 0.0:
                    # 왜곡 보정된 정규화 좌표로 역투영 (solvePnP와 동일한 카메라 모델)
                    und = cv2.undistortPoints(
                        np.array([[[cxf, cyf]]], dtype=np.float32),
                        self.camera_matrix, self.dist_coeffs
                    ).flatten()
                    P_cam = np.array([und[0] * z_val, und[1] * z_val, z_val, 1.0])

                    T_base2cam = np.linalg.inv(T_cam2base)
                    P_base = T_base2cam @ P_cam

                    # 날것의 데이터(Raw Data)
                    raw_x = P_base[0] * 100.0
                    raw_y = P_base[1] * 100.0
                    raw_z = P_base[2] * 100.0

                    # ========================================================
                    # [필터링] 5-프레임 중간값(Median) 필터 적용 (여기서 history 생성!)
                    # ========================================================
                    if not hasattr(self, 'history_x'):
                        from collections import deque
                        self.history_x = deque(maxlen=5)
                        self.history_y = deque(maxlen=5)
                        self.history_z = deque(maxlen=5)

                    self.history_x.append(raw_x)
                    self.history_y.append(raw_y)
                    self.history_z.append(raw_z)

                    rel_x = np.median(self.history_x)
                    rel_y = np.median(self.history_y)
                    rel_z = np.median(self.history_z)

                    # [스페이스바용] 맨 아래 waitKey 로직에서 쓸 수 있게 변수에 저장
                    self.current_filtered_pos = (rel_x, rel_y, rel_z)
                    self.current_P_cam = P_cam[:3] * 100.0  # 카메라 좌표계 (cm)
                    self.last_meas_time = time.time()

                    # 측정 중이면 raw 값 수집, 30개 모이면 통계 출력
                    if self.sample_buf is not None:
                        self.sample_buf.append((raw_x, raw_y, raw_z))
                        if len(self.sample_buf) >= 30:
                            self.finish_measurement()

                    # (1초마다 출력하던 부분은 여기서 깔끔하게 삭제되었습니다!)

                    # 시각화 (필터링된 부드러운 값으로 화면에 표시)
                    cv2.circle(img, (cx, cy), 6, (0, 255, 0), -1)
                    coord_text = f"X:{rel_x:.1f} Y:{rel_y:.1f} Z:{rel_z:.1f}cm"
                    cv2.putText(img, coord_text, (cx + 15, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

                    
        cv2.imshow("Real Target Tracker", img)
        key = cv2.waitKey(1) & 0xFF

        # 1. 'q' 또는 'ESC' 키: 안전하게 종료
        if key == ord('q') or key == 27:
            self.get_logger().info("종료 키가 입력되었습니다. 노드를 안전하게 끕니다.")
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            raise KeyboardInterrupt

        # 2. '스페이스바(32)' 키: 30프레임 수집 측정 시작
        elif key == 32:
            if hasattr(self, 'current_filtered_pos'):
                if self.sample_buf is None:
                    self.sample_buf = []
                    self.get_logger().info("measuring... (30 frames, 카메라 고정 유지)")
            else:
                self.get_logger().info("target not found.")

        # 3. 창의 'X' 버튼: 안전하게 종료
        if cv2.getWindowProperty("Real Target Tracker", cv2.WND_PROP_AUTOSIZE) < 0:
            self.get_logger().info("window closed.")
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            raise KeyboardInterrupt

def main(args=None):
    rclpy.init(args=args)
    node = RobustDepthArucoTracker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()