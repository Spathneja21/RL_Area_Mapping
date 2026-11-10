"""Gymnasium env: diff-drive AMR with a 360-degree lidar exploring an unknown map.

Truth = a saved ROS map from AMR_Practice_project. Reward = newly mapped area (m^2).
"""
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from mapping import RES, CROP, UNKNOWN, FREE, load_truth, dilate, flood, cast, integrate, observe

MAPS_DIR = os.environ.get('AMR_MAPS', os.path.expanduser('~/projects/AMR_Practice_project/src/amr_bot/maps'))
# office = Clearpath office (~42 x 23 m), market = supermarket (~12 x 8 m)
MAPS = {'office': 'office.yaml', 'market': 'market.yaml'}
DT = 1.0                                   # s per decision
# (v m/s, w rad/s) — same limits as lidar_explorer (v_max 0.2, w_max 0.8)
ACTIONS = np.array([[0.2, 0.0], [0.2, 0.6], [0.2, -0.6], [0.0, 0.8], [0.0, -0.8]])
ROBOT_R = 0.15
MARGIN = 0.05


class _World:
    def __init__(self, yaml_path):
        self.occ, self.origin = load_truth(yaml_path)
        self.blocked = dilate(self.occ, int(np.ceil((ROBOT_R + MARGIN) / RES)))
        # spawn region = largest connected free component of the inflated map
        free = ~self.blocked
        best = None
        left = free.copy()
        while left.sum() > 0:
            seed = tuple(np.argwhere(left)[0])
            comp = flood(free, seed)
            left &= ~comp
            if best is None or comp.sum() > best.sum():
                best = comp
        self.spawn = np.argwhere(best)
        # cells a perfect explorer can map: free cells connected to the reachable area (un-inflated)
        self.mappable = flood(~self.occ, tuple(self.spawn[0])).sum()


class ExploreEnv(gym.Env):
    def __init__(self, maps=('office', 'market'), max_steps=600, seed=None):
        self.worlds = [_World(os.path.join(MAPS_DIR, MAPS[m])) for m in maps]
        self.max_steps = max_steps
        self.observation_space = spaces.Box(0, 255, (6, CROP, CROP), np.uint8)
        self.action_space = spaces.Discrete(len(ACTIONS))
        self.np_random = np.random.default_rng(seed)

    def _sense(self):
        w = self.w
        before = self.known
        integrate(self.belief, w.origin, self.pose, *cast(w.occ, w.origin, self.pose))
        self.known = int((self.belief != UNKNOWN).sum())
        r, c = self._cell(self.pose)
        self.trail[r, c] = True
        return (self.known - before) * RES * RES

    def _cell(self, pose):
        return (int((pose[1] - self.w.origin[1]) / RES), int((pose[0] - self.w.origin[0]) / RES))

    def _obs(self):
        return observe(self.belief, self.trail, self.w.origin, self.pose)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        self.w = self.worlds[self.np_random.integers(len(self.worlds))]
        r, c = self.w.spawn[self.np_random.integers(len(self.w.spawn))]
        if options and 'pose' in options:
            self.pose = np.array(options['pose'], float)
        else:
            self.pose = np.array([(c + 0.5) * RES + self.w.origin[0], (r + 0.5) * RES + self.w.origin[1],
                                  self.np_random.uniform(-np.pi, np.pi)])
        self.belief = np.full(self.w.occ.shape, UNKNOWN, np.int8)
        self.trail = np.zeros(self.w.occ.shape, bool)
        self.known, self.steps, self.collisions = 0, 0, 0
        self._sense()
        return self._obs(), {}

    def coverage(self):
        return float((self.belief == FREE).sum()) / self.w.mappable

    def step(self, a):
        v, wz = ACTIONS[a] * self.np_random.uniform(0.9, 1.1, 2)   # 10% actuation noise
        collided = False
        n = 10
        for _ in range(n):
            x, y, th = self.pose
            nxt = np.array([x + v * np.cos(th) * DT / n, y + v * np.sin(th) * DT / n, th + wz * DT / n])
            r, c = self._cell(nxt)
            if not (0 <= r < self.w.occ.shape[0] and 0 <= c < self.w.occ.shape[1]) or self.w.blocked[r, c]:
                collided = True
                nxt[:2] = self.pose[:2]              # blocked: translation stops, rotation still applies
            self.pose = nxt
        self.pose[2] = (self.pose[2] + np.pi) % (2 * np.pi) - np.pi
        reward = self._sense() - 0.02 - 0.5 * collided
        self.collisions += collided
        self.steps += 1
        done = self.coverage() > 0.95
        trunc = self.steps >= self.max_steps
        return self._obs(), reward, done, trunc, {'coverage': self.coverage(), 'collisions': self.collisions}


if __name__ == '__main__':   # self-check: spawn, sensing, collision blocking, obs shape
    import time
    env = ExploreEnv(seed=0)
    for w, name in zip(env.worlds, MAPS):
        print(name, w.occ.shape, 'spawn cells', len(w.spawn), 'mappable m^2', w.mappable * RES * RES)
    obs, _ = env.reset(seed=1)
    assert obs.shape == (6, CROP, CROP) and obs[0].any(), 'robot must see free space at spawn'
    r0 = env.known
    t = time.time()
    total, cov = 0, []
    for i in range(300):
        obs, rew, done, trunc, info = env.step(env.action_space.sample())
        total += rew
        r, c = env._cell(env.pose)
        assert not env.w.blocked[r, c], 'robot ended inside an obstacle'
    print(f'300 random steps: {(time.time() - t) / 300 * 1e3:.2f} ms/step, return {total:.1f}, '
          f'coverage {info["coverage"]:.2f}, collisions {info["collisions"]}')
    assert env.known > r0
    print('ok')
