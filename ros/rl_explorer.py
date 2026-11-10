#!/usr/bin/env python3
"""ROS 2 node: drives the AMR with the trained PPO policy (ONNX) while slam_toolbox builds the map.

The policy's input is the node's own lidar map, built from /scan with the same integrate() as training
and placed with SLAM's map->base_link pose. (slam_toolbox's own /map starts as a ~3 m^2 sparse blob and
only grows once the robot translates, which the policy never saw in training.) /map is only logged.
Run (sourced ROS shell):  python3 rl_explorer.py --ros-args -p model:=/rl/models/ppo_explore.onnx
"""
import os
import sys
import time
import numpy as np
import onnxruntime as ort
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.signals import SignalHandlerOptions
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from geometry_msgs.msg import Twist
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformListener

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'rl'))
from mapping import RES, UNKNOWN, integrate, observe  # noqa: E402

DT = 1.0
ACTIONS = np.array([[0.2, 0.0], [0.2, 0.6], [0.2, -0.6], [0.0, 0.8], [0.0, -0.8]])   # must match rl/env.py
HALF = 40.0                                     # belief covers +-40 m around the map origin


class RLExplorer(Node):
    def __init__(self):
        super().__init__('rl_explorer')
        model = self.declare_parameter('model', '/rl/models/ppo_explore.onnx').value
        self.max_time = self.declare_parameter('max_time_s', 900.0).value
        self.stop_dist = self.declare_parameter('stop_dist', 0.25).value   # lidar guard, front +-30 deg
        self.sess = ort.InferenceSession(model)
        n = int(2 * HALF / RES)
        self.origin = np.array([-HALF, -HALF])
        self.belief = np.full((n, n), UNKNOWN, np.int8)
        self.trail = np.zeros((n, n), bool)
        self.cmd = np.zeros(2)
        self.scan = None
        self.slam_area = 0.0
        self.t0 = None
        self.last_log = 0.0
        self.tf = Buffer()
        TransformListener(self.tf, self)
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(OccupancyGrid, '/map', self.on_map, qos)
        self.create_subscription(LaserScan, '/scan', lambda s: setattr(self, 'scan', s), 10)
        self.create_timer(DT, self.decide)
        self.create_timer(0.1, self.drive)
        self.get_logger().info(f'loaded {model}')

    def on_map(self, m):
        self.slam_area = (np.asarray(m.data) != -1).sum() * m.info.resolution ** 2   # same metric as lidar_explorer

    def pose(self):
        try:
            t = self.tf.lookup_transform('map', 'base_link', rclpy.time.Time())
        except Exception as e:  # TF not up yet
            self.get_logger().warn(f'no map->base_link yet: {e}', throttle_duration_sec=5.0)
            return None
        q = t.transform.rotation
        yaw = np.arctan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        return np.array([t.transform.translation.x, t.transform.translation.y, yaw])

    def decide(self):
        p = self.pose()
        if p is None or self.scan is None:
            return
        now = time.monotonic()
        self.t0 = self.t0 or now
        if now - self.last_log >= 10.0:
            self.last_log = now
            self.get_logger().info(f'mapped area: {self.slam_area:.1f} m^2  (t={now - self.t0:.0f} s)')
        if now - self.t0 > self.max_time:
            self.cmd[:] = 0
            self.get_logger().info(f'exploration finished: time budget used ({self.slam_area:.1f} m^2)', once=True)
            return
        s = self.scan
        angles = s.angle_min + s.angle_increment * np.arange(len(s.ranges))
        ranges = np.asarray(s.ranges, float)
        ranges[ranges < s.range_min] = np.nan                      # too close to trust: drop the beam
        keep = ~np.isnan(ranges)
        integrate(self.belief, self.origin, p, angles[keep], ranges[keep])
        r, c = np.floor((p[1::-1] - self.origin[::-1]) / RES).astype(int)
        self.trail[r, c] = True
        obs = observe(self.belief, self.trail, self.origin, p)
        self.cmd = ACTIONS[int(np.argmax(self.sess.run(None, {'obs': obs[None]})[0]))].copy()

    def drive(self):
        v, w = self.cmd
        if v > 0 and self.scan is not None:                         # guard: never push into something
            r = np.asarray(self.scan.ranges)
            mid = len(r) // 2                                       # index 180 = straight ahead
            f = r[mid - 30:mid + 31]
            if (np.isfinite(f) & (f > self.scan.range_min) & (f < self.stop_dist)).any():
                v, w = 0.0, w or 0.8
        msg = Twist()
        msg.linear.x, msg.angular.z = float(v), float(w)
        self.pub.publish(msg)


def main():
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)     # keep the context alive to send a stop
    node = RLExplorer()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    node.pub.publish(Twist())
    rclpy.shutdown()


if __name__ == '__main__':
    main()
