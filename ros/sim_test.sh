#!/usr/bin/env bash
# Headless end-to-end run in the container: MuJoCo office + slam_toolbox + an explorer, then save the map.
# usage (inside container): /rl/ros/sim_test.sh rl|frontier SECONDS [model.onnx]

EXPLORER=$1; SECS=$2; MODEL=${3:-/rl/models/ppo_explore.onnx}
OUT=/rl/runs/sim_${EXPLORER}_$(date +%H%M%S); mkdir -p "$OUT"
source /opt/ros/humble/setup.bash; source /ros2_ws/install/setup.bash
export LIBGL_ALWAYS_SOFTWARE=1; unset DISPLAY                 # no window: bridge runs without its viewer

ros2 run amr_bot mujoco_bridge --ros-args -p scene:=/ros2_ws/src/amr_bot/models/world/office_scene.xml \
  > "$OUT/bridge.log" 2>&1 &
ros2 launch amr_bot rsp.launch.py > "$OUT/rsp.log" 2>&1 &
ros2 run slam_toolbox async_slam_toolbox_node --ros-args \
  --params-file /ros2_ws/src/amr_bot/config/async_slam_toolbox_cfg.yaml > "$OUT/slam.log" 2>&1 &
sleep 8
if [ "$EXPLORER" = rl ]; then
  python3 /rl/ros/rl_explorer.py --ros-args -p model:="$MODEL" -p max_time_s:="$SECS.0" > "$OUT/explorer.log" 2>&1 &
else
  ros2 run amr_bot lidar_explorer > "$OUT/explorer.log" 2>&1 &
fi
sleep "$SECS"
ros2 run nav2_map_server map_saver_cli -f "$OUT/map" > "$OUT/map_saver.log" 2>&1
pkill -INT -f rl_explorer.py; pkill -INT -f lidar_explorer; pkill -INT -f async_slam_toolbox_node
pkill -INT -f robot_state_publisher; pkill -INT -f mujoco_bridge; sleep 2
pkill -9 -f "mujoco_bridge|async_slam_toolbox_node|robot_state_publisher|rl_explorer.py|lidar_explorer|ros2 launch"
grep -E "mapped area|complete|finished" "$OUT/explorer.log" | tail -3
chown -R "$(stat -c %u:%g /rl)" "$OUT"
echo "output: $OUT"
