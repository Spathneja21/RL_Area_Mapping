#!/usr/bin/env python3
"""Debug helper: print arrival time and known area of every /map message."""
import time
import numpy as np
import rclpy
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from nav_msgs.msg import OccupancyGrid

rclpy.init()
n = rclpy.create_node('map_probe')
t0 = time.monotonic()
qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
n.create_subscription(OccupancyGrid, '/map', lambda m: print(
    f't={time.monotonic() - t0:5.1f}s known {(np.asarray(m.data) != -1).sum() * m.info.resolution ** 2:6.1f} m^2',
    flush=True), qos)
rclpy.spin(n)
