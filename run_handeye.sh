#!/usr/bin/env bash
# Run handeye_calib.py inside the EXISTING ros2_dobot container.
#
# What this deliberately does NOT do:
#   - install anything, anywhere (the container already has cv2 4.5.4, numpy,
#     rclpy, cv_bridge and realsense2_camera -- checked, nothing is missing)
#   - create, recreate, restart or reconfigure any container
#   - touch ~/venv_ammr, ~/ammr or ~/dobot_ws
#
# ~/realsense-ros is not bind-mounted into the container, so the script is
# copied in with `docker cp` and the JSON/PNG it writes is copied back out.
# `docker cp` needs no container change and no restart.
#
# Run on the JETSON HOST, from anywhere:
#     ~/realsense-ros/run_handeye.sh selftest                  # no hardware
#     ~/realsense-ros/run_handeye.sh gen-board --squares 9x6 --square-mm 20
#     ~/realsense-ros/run_handeye.sh collect  --squares 9x6 --square-mm 20
#     ~/realsense-ros/run_handeye.sh solve
#     ~/realsense-ros/run_handeye.sh verify
#
# A first argument ending in .py runs THAT script in the container instead,
# for the one-off diagnostics that live next to this one. HANDEYE_PY picks a
# different interpreter -- probe_euler.py needs pinocchio, which lives in the
# workspace venv rather than the container's system python:
#     HANDEYE_PY=/root/dobot_ws/.venv/bin/python3 \
#       ~/realsense-ros/run_handeye.sh probe_euler.py 192.168.5.1
#
# collect/verify need the camera driver up in its own terminal:
#     docker exec -it ros2_dobot bash -lc "source /opt/ros/humble/setup.bash && \
#       ros2 launch realsense2_camera rs_launch.py camera_name:=d405 align_depth.enable:=true"
set -eu

HERE="$(cd "$(dirname "$0")" && pwd)"
CTR=ros2_dobot
WORK=/tmp/handeye
# Files that live on the host side and must survive between invocations.
STATE="handeye_samples.json handeye_result.json board.png"

if ! docker inspect -f '{{.State.Running}}' "$CTR" 2>/dev/null | grep -q true; then
    echo "container $CTR is not running" >&2
    exit 1
fi

SCRIPT=handeye_calib.py
case "${1:-}" in
    *.py) SCRIPT="$1"; shift ;;
esac
[ -f "$HERE/$SCRIPT" ] || { echo "no such script: $HERE/$SCRIPT" >&2; exit 1; }

docker exec "$CTR" mkdir -p "$WORK"
docker cp "$HERE/$SCRIPT" "$CTR:$WORK/$SCRIPT"
# The diagnostics import handeye_calib for the target model and the
# tool_vector convention, so it has to be the current one -- without this an
# edit to it is invisible and the container silently runs the previous copy.
[ "$SCRIPT" = handeye_calib.py ] || \
    docker cp "$HERE/handeye_calib.py" "$CTR:$WORK/handeye_calib.py"
for f in $STATE; do
    [ -f "$HERE/$f" ] && docker cp "$HERE/$f" "$CTR:$WORK/$f"
done

TTY_FLAGS=""
[ -t 0 ] && TTY_FLAGS="-it"          # a TTY only when there is one to hand over

# The container was created with DISPLAY=:1 baked in, but the host's X server is
# :0 now, so the stale value makes every cv2.imshow die with "Can't initialize
# GTK backend". Take the display from the socket that actually exists; an
# already-set DISPLAY (running from the desktop) wins.
if [ -z "${DISPLAY:-}" ]; then
    sock="$(ls /tmp/.X11-unix/ 2>/dev/null | head -1)"
    [ -n "$sock" ] && DISPLAY=":${sock#X}"
fi

# Passing the arguments after the `_` matters: `bash -c 'script' arg` makes arg
# $0, not $1, so a bare `bash -lc "..." solve` silently drops the subcommand.
rc=0
docker exec $TTY_FLAGS ${DISPLAY:+-e DISPLAY="$DISPLAY"} "$CTR" bash -lc '
source /opt/ros/humble/setup.bash
source /root/dobot_ws/install/setup.bash 2>/dev/null || true
cd '"$WORK"'
exec '"${HANDEYE_PY:-python3}"' '"$SCRIPT"' "$@"
' _ "$@" || rc=$?

# Always copy results back, even on a non-zero exit -- a run that crashed after
# capturing 12 poses still has 12 poses worth keeping.
for f in $STATE; do
    docker cp "$CTR:$WORK/$f" "$HERE/$f" 2>/dev/null || true
done

exit "$rc"
