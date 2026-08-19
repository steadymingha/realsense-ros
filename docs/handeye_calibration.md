# Hand-eye 캘리브레이션 절차 (D405 ↔ CR7 플랜지)

카메라를 떼거나 옮기면 반드시 다시 해야 한다. `T_flange_cam`은 플랜지와 카메라 사이의
물리적 관계 그 자체라, 마운트가 바뀌면 이전 값은 무효다. 소요 10~15분.

**결과물:** `~/realsense-ros/handeye_result.json` 의 `T_flange_cam` (4x4).
하류 코드는 이 파일을 읽어 쓰므로, 재캘리는 파일 교체 한 번이고 코드는 안 건드린다.

```
P_base = T_base_flange @ T_flange_cam @ P_cam
```

---

## 0. 준비

- **보드**: calib.io ChArUco `8x11`, 체커 15 mm, 마커 11 mm, `DICT_4X4`
  → 도구 기본값이라 인자 없이 그냥 돌리면 된다.
  다른 보드면 `--squares 11x8 --square-mm 15 --marker-mm 11 --dict DICT_4X4_50`.
- **보드를 움직이지 않게 고정.** 테이프든 뭐든. 이게 이 절차에서 제일 중요하다.
- 펜던트에서 **user=0, tool=0** 확인. 다른 tool이 잡혀 있으면 "플랜지"가 플랜지가 아니다.
- 로봇 전원 ON (조그해야 하므로). 브링업은 있어도 없어도 된다 —
  이 도구는 로봇에 아무것도 보내지 않고 실시간 피드(30004)를 읽기만 한다.

## 1. 카메라 드라이버 (별도 터미널, 캘리 내내 켜둘 것)

```bash
docker exec -it ros2_dobot bash -lc "source /opt/ros/humble/setup.bash && \
  ros2 launch realsense2_camera rs_launch.py camera_name:=d405 align_depth.enable:=true"
```

해상도는 기본값(848x480)으로 둔다. hand-eye 결과는 물리적 변환이라 해상도와 무관하고,
1280x720은 이 리그에서 프레임 타임아웃이 났다.

## 2. 이전 결과 보관

```bash
cd ~/realsense-ros
mv handeye_samples.json handeye_samples_$(date +%m%d).json
mv handeye_result.json  handeye_result_$(date +%m%d).json
```

새 마운트가 더 나쁠 수도 있으니 되돌릴 길을 남긴다.

## 3. 수집

```bash
./run_handeye.sh collect --auto --fresh
```

**`--fresh` 필수.** 빼먹으면 옛 마운트의 샘플에 새 샘플이 이어붙고, 두 마운트가 섞인
데이터로 자신 있게 틀린 답이 나온다.

젯슨 모니터에 창이 뜬다. 자동 모드라 **키를 누를 필요 없다** — 조그해서 멈추면 0.8초 뒤
알아서 찍는다. 이미 찍은 자세와 비슷하면(3 cm / 8° 이내) 건너뛴다.

**사람이 할 일 — 15~20자세:**

- **보드가 화면에 꽉 차게.** 아래 참고 — 개수보다 이게 훨씬 중요하다
- 보드까지 **20~40 cm**
- **손목을 두 축 이상으로 ±30° 넘게** 돌린다. 위치만 옮기면 hand-eye는 수학적으로
  퇴화해서 그럴듯한 오답이 나온다
- 거리도 같이 바꾼다
- 팔이 멀어져 보드가 안 보이면 **보드를 옮기지 말고 팔을 되돌린다**
- 실제로 탐지에 쓸 영역 근처에서도 몇 자세 잡는다

캡처마다 콘솔에 `rotation spread`가 찍힌다. **40° 넘으면** 충분하다.

> ### 개수 채우려 하지 말 것 — 뷰 품질이 정확도를 만든다
>
> 실측 (2026-08-07 vs 08-11, 같은 리그):
>
> | | 48 샘플 | 35 샘플 |
> |---|---|---|
> | 뷰당 코너 (median, 최대 70) | 19 | **69** |
> | solve spread | 2.39 mm | **1.69 mm** |
> | verify p95 | 10.04 mm | **6.2 mm** |
>
> **샘플을 13개 줄이고 오히려 좋아졌다.** 48개짜리는 팔을 크게 휘두르느라 보드가
> 화면 밖으로 반쯤 나간 뷰가 많았고(코너 8~19개), 35개짜리는 보드를 프레임에 꽉
> 채웠다. 코너가 적으면 보드의 좁은 조각만 보는 셈이라 자세 추정이 불안정하고,
> 평면 미러 해로 넘어갈 확률도 올라간다.
>
> 캡처 로그의 `N/70 inliers`를 보면서, **50 이상**이 꾸준히 나오는 자세로 잡을 것.
> 38개 미만이 자주 뜨면 보드에 더 다가가거나 팔을 덜 벌린다.
> 좋은 뷰 35개면 4~5분이면 끝난다.
캡처할 때마다 파일에 저장되므로 중간에 끊겨도 잃지 않는다. `Q`로 종료.

원격에서 화면이 안 보일 때: `docker exec ros2_dobot tail -f /tmp/handeye/collect.log`

## 4. 풀기

```bash
./run_handeye.sh solve
```

출력에서 볼 것:

```
rigidity: all samples consistent with one fixed setup
rotation spread across poses: 48.0 deg
  TSAI            2.48 mm     0.70 deg
  PARK            2.39 mm     0.70 deg
  ...
best: PARK   board position spread 2.39 mm rms
  xyz (m)  = [-0.00649, -0.08428, +0.02321]   |xyz| = 87.7 mm
```

- `rigidity`에서 **"only N/M samples belong to one rigid setup"** 이 뜨면 중간에 뭔가
  움직인 것이다. 보드를 안 건드렸다면 **마운트가 헐거운 것** — 값을 쓰지 말고
  마운트부터 조인다.
- 5개 방법의 `|xyz|`가 서로 비슷해야 한다. 제각각이면 회전 다양성이 부족한 것.

## 5. 검증

```bash
./run_handeye.sh verify
```

보드 좌표를 base_link로 계속 출력한다. 보드는 안 움직이니 **팔을 어떻게 움직이든 이 값이
그대로여야** 한다. 조그하면서 지켜보고 `Q`로 종료하면 판정이 나온다.

```
PASS: board holds to 4.8 mm (p95 of inliers) over 210 mm / 45 deg of arm motion.
```

- **PASS** = p95 < 10 mm. 이 값을 쓰면 된다
- **INCONCLUSIVE** = 팔을 100 mm / 30° 이상 움직여야 판정이 나온다. 더 조그할 것
- **FAIL** = p95 ≥ 10 mm. 쓰지 말 것

이상치가 몇 % 뜨는 건 정상이다 — 평면 타겟의 미러 해(mirror pose)라 캘리 오차가 아니다.
다만 이상치 비율이 크면 문제다.

---

## 자주 걸리는 것

| 증상 | 원인 / 조치 |
|---|---|
| `Can't initialize GTK backend` | 컨테이너에 `DISPLAY=:1`이 박혀 있는데 호스트 X는 `:0`. `run_handeye.sh`가 자동으로 잡지만, 직접 `docker exec` 할 땐 `-e DISPLAY=:0` |
| 카메라 프레임 안 들어옴 | 드라이버를 강제 종료하면 USB가 물린다. `pkill` 후 **8초 이상 기다렸다가** 재실행 |
| `numpy 1.x cannot be run in NumPy 2.2.6` | 워크스페이스 venv(numpy 2)와 시스템 cv2(numpy 1)를 한 프로세스에서 쓴 것. 캘리 도구는 시스템 python3로 돌아간다(기본값). pinocchio가 필요한 스크립트만 `HANDEYE_PY=/root/dobot_ws/.venv/bin/python3` |
| 보드가 검출은 되는데 값이 이상 | 체커 실측값 확인. 인쇄 배율이 5% 틀리면 모든 거리가 5% 틀린다. 재수집 없이 `solve --square-mm 14.8` 로 다시 풀 수 있다 |
| solve가 20 mm 넘는 spread | 회전 부족(가장 흔함) → 손목 더 돌려서 재수집 |

## 파일

| 파일 | 역할 |
|---|---|
| `handeye_calib.py` | 도구 본체 (`collect` / `solve` / `verify` / `gen-board` / `selftest`) |
| `run_handeye.sh` | `ros2_dobot` 컨테이너에서 실행하는 래퍼. `docker cp`로 넣고 결과를 꺼낸다 |
| `handeye_result.json` | **결과.** `T_flange_cam` + K/dist + 보드 사양 |
| `handeye_samples.json` | 원본 관측(관절각·TCP·코너 픽셀). 재수집 없이 재계산 가능 |
| `probe_board.py` | 처음 보는 보드의 딕셔너리·레이아웃 판별 |
| `probe_euler.py` | 펌웨어/tool 프레임 바뀌었을 때 TCP 자세 규약 재검증 |
| `diag_rigid.py` | 샘플별 상호일치 히스토그램. 마운트 강성 의심될 때 |
| `test_verify_summary.py` | verify 리포트 경로 테스트 (하드웨어 불필요) |

하드웨어 없이 도구 자체 점검: `./run_handeye.sh selftest`

## 그리퍼 장착 때 같이 손볼 것

- `TCP_OFFSET_M` (~120 mm) — 지금은 없는 그리퍼를 가정한 값. 실측값으로 교체
- URDF `d405_joint` — 부모가 `gripper_base_link`. 캘리 실측값으로 교체
- 충돌 모델에 그리퍼 형상 추가, 주석 처리된 카메라 collision 복원 검토
- 마운트는 **반복 장착 가능한 구조**(핀+볼트, 위치결정 홈)로. 떼었다 붙일 때마다
  재캘리하는 것과 값이 유지되는 것은 나중에 크게 차이난다
