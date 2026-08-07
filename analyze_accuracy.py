"""measurements.csv에서 트래킹 반복 정밀도(repeatability) 통계를 산출한다.

전제: base 마커와 타겟은 고정, 카메라만 여러 시점으로 옮기며 측정한 세션.
참값 = 전체 측정의 평균으로 두고, 각 측정의 편차로 오차 통계를 계산한다.

사용법:
    측정 세션 후:  python3 analyze_accuracy.py [measurements.csv]
    새 세션 시작 전에는 기존 csv를 지우거나 백업할 것 (측정이 계속 append됨)
"""
import sys
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "measurements.csv"
data = np.genfromtxt(path, delimiter=",", names=True)
if data.shape == ():  # 행이 1개면 0-d로 읽힘
    print("측정이 1개뿐입니다. 여러 시점에서 20개 이상 모으세요.")
    sys.exit(1)

pos = np.column_stack([data["mean_x"], data["mean_y"], data["mean_z"]])
base_d = np.sqrt(data["base_tx"]**2 + data["base_ty"]**2 + data["base_tz"]**2)
n = len(pos)

dev = pos - pos.mean(axis=0)          # 전체 평균 기준 편차 (cm)
r3d = np.linalg.norm(dev, axis=1)     # 측정별 3D 오차 (cm)

print(f"측정 수          : {n}")
print(f"base 거리 범위   : {base_d.min():.0f} ~ {base_d.max():.0f} cm")
print()
print("축별 오차 (전체 평균 기준, cm)")
print(f"  {'축':<2} {'RMS(=std)':>9} {'|편차|중간값':>10} {'95%':>6} {'최대':>6}")
for i, ax in enumerate("XYZ"):
    d = np.abs(dev[:, i])
    print(f"  {ax:<2} {dev[:, i].std():>9.2f} {np.median(d):>10.2f} "
          f"{np.percentile(d, 95):>6.2f} {d.max():>6.2f}")
print()
print(f"3D RMS 오차      : {np.sqrt((r3d**2).mean()):.2f} cm")
print(f"3D 오차 중간값   : {np.median(r3d):.2f} cm")
print(f"3D 오차 95퍼센타일: {np.percentile(r3d, 95):.2f} cm")
print(f"3D 오차 최대     : {r3d.max():.2f} cm")
print()
print("인용 예시: "
      f"\"고정 타겟을 {n}개 시점(base 거리 {base_d.min():.0f}–{base_d.max():.0f}cm)에서 "
      f"측정한 결과, 3D RMS 오차 {np.sqrt((r3d**2).mean()):.2f}cm, "
      f"95%가 {np.percentile(r3d, 95):.2f}cm 이내\"")
