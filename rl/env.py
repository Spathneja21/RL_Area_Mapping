"""Gymnasium env: diff-drive AMR with a 360-degree lidar exploring an unknown map.

Truth = a saved ROS map from AMR_Practice_project. Reward = newly mapped area (m^2).
"""
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from mapping import RES, CROP, N_CHANNELS, UNKNOWN, FREE, load_truth, dilate, flood, cast, integrate, observe
from frontier import K, N_FEAT, REACHED, plan, candidates, is_frontier, path_to, steer

MAPS_DIR = os.environ.get('AMR_MAPS', os.path.expanduser('~/projects/AMR_Practice_project/src/amr_bot/maps'))
# office = Clearpath office (~42 x 23 m), market = supermarket (~12 x 8 m)
MAPS = {'office': 'office.yaml', 'market': 'market.yaml'}
DT = 1.0                                   # s per decision
# (v m/s, w rad/s) — same limits as lidar_explorer (v_max 0.2, w_max 0.8)
ACTIONS = np.array([[0.2, 0.0], [0.2, 0.6], [0.2, -0.6], [0.0, 0.8], [0.0, -0.8]])
FRONT = np.radians(np.arange(-30, 31, 5))  # lidar guard: beams within +-30 deg of straight ahead
STOP_DIST = 0.25                           # m, = rl_explorer stop_dist
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
    def __init__(self, maps=('office', 'market'), max_steps=1000, seed=None, pre_explore=0.0, guard=True):
        self.worlds = [_World(os.path.join(MAPS_DIR, MAPS[m])) for m in maps]
        self.max_steps = max_steps
        self.guard = guard
        self.pre_explore = pre_explore             # P(episode starts with the area around the spawn mapped)
        self.observation_space = spaces.Box(0, 255, (N_CHANNELS, CROP, CROP), np.uint8)
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
        if self.np_random.random() < self.pre_explore:
            # Mid-exploration start: scans from 2-12 visited spots within 3-15 m are already mapped, so
            # "everything near me is done, go elsewhere" (the hard part of exploring) is common in training.
            near = self.w.spawn[np.hypot(*(self.w.spawn - (r, c)).T) * RES < self.np_random.uniform(3, 15)]
            for rr, cc in near[self.np_random.integers(len(near), size=self.np_random.integers(2, 13))]:
                p = np.array([(cc + 0.5) * RES + self.w.origin[0], (rr + 0.5) * RES + self.w.origin[1], 0.0])
                integrate(self.belief, self.w.origin, p, *cast(self.w.occ, self.w.origin, p))
                self.trail[rr, cc] = True
            self.known = int((self.belief != UNKNOWN).sum())
        self._sense()
        self.cov_log = [self.coverage()]                # coverage per second, for evaluation curves
        self.traj = [self.pose[:2].copy()]
        return self._obs(), {}

    def coverage(self):
        return float((self.belief == FREE).sum()) / self.w.mappable

    def step(self, a):
        reward = self._move(a)
        done = self.coverage() > 0.95
        trunc = self.steps >= self.max_steps
        return self._obs(), reward, done, trunc, {'coverage': self.coverage(), 'collisions': self.collisions}

    def _move(self, a):
        """One 1 s motion primitive + sensing. Returns its reward; logs coverage once per second."""
        v, wz = ACTIONS[a] * self.np_random.uniform(0.9, 1.1, 2)   # 10% actuation noise
        collided = False
        n = 10
        for _ in range(n):
            if self.guard and v > 0 and cast(self.w.occ, self.w.origin, self.pose, FRONT)[1].min() < STOP_DIST:
                v, wz = 0.0, wz or 0.8             # same lidar guard as ros/rl_explorer.py: rotate, don't push
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
        self.cov_log.append(self.coverage())
        self.traj.append(self.pose[:2].copy())
        return reward


class MacroExploreEnv(ExploreEnv):
    """Hierarchical version: an action picks one of the K nearest frontier candidates; a BFS path follower
    drives there with the 1 s primitives. One step lasts until the goal is reached or seen (no longer a
    frontier), 60 s pass, or 3 bumps. Reward = sum of the 1 s rewards. Slot 0 = greedy nearest frontier."""

    def __init__(self, max_macro_s=60, **kw):
        super().__init__(**kw)
        self.max_macro_s = max_macro_s
        self.observation_space = spaces.Dict({
            'map': spaces.Box(0, 255, (N_CHANNELS, CROP, CROP), np.uint8),
            'cands': spaces.Box(-np.inf, np.inf, (K, N_FEAT), np.float32)})
        self.action_space = spaces.Discrete(K)

    def _macro_obs(self):
        for _ in range(2):
            self.dist = plan(self.belief, self.bumps, self._cell(self.pose))
            self.goals, feats = candidates(self.belief, self.dist, self.pose, self.w.origin)
            if len(self.goals) or not self.bumps.any():
                break
            self.bumps[:] = False                         # nothing left: forget bumps/blacklist once and retry
        return {'map': self._obs(), 'cands': feats}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.bumps = np.zeros(self.w.occ.shape, bool)
        return self._macro_obs(), {}

    def step(self, a):
        reward = 0.0
        if len(self.goals):
            goal = self.goals[min(int(a), len(self.goals) - 1)]
            path = path_to(self.dist, goal)
            coll0, known0 = self.collisions, self.known
            for _ in range(self.max_macro_s):
                before = self.collisions
                reward += self._move(steer(self.pose, path, self.w.origin))
                if self.collisions > before:              # bumper: remember the cell ahead as an obstacle
                    x, y, th = self.pose
                    r, c = self._cell((x + 0.25 * np.cos(th), y + 0.25 * np.sin(th), th))
                    self.bumps[np.clip(r, 0, self.bumps.shape[0] - 1), np.clip(c, 0, self.bumps.shape[1] - 1)] = True
                r, c = self._cell(self.pose)
                if (self.steps >= self.max_steps or self.coverage() > 0.95 or self.collisions - coll0 >= 3
                        or abs(r - goal[0]) + abs(c - goal[1]) <= REACHED or not is_frontier(self.belief, goal)):
                    break
            if self.known == known0 and is_frontier(self.belief, goal):
                self.bumps[tuple(goal)] = True            # no progress toward it: blacklist (as lidar_explorer)
        obs = self._macro_obs()
        done = self.coverage() > 0.95 or not len(self.goals)  # no reachable frontier left = finished
        trunc = self.steps >= self.max_steps
        return obs, reward, done, trunc, {'coverage': self.coverage(), 'collisions': self.collisions}


if __name__ == '__main__':   # self-check: spawn, sensing, collision blocking, obs shape
    import time
    env = ExploreEnv(seed=0)
    for w, name in zip(env.worlds, MAPS):
        print(name, w.occ.shape, 'spawn cells', len(w.spawn), 'mappable m^2', w.mappable * RES * RES)
    obs, _ = env.reset(seed=1)
    assert obs.shape == (N_CHANNELS, CROP, CROP) and obs[0].any(), 'robot must see free space at spawn'
    assert obs[3].any() or obs[7].any(), 'a fresh map must have frontiers'
    pre = ExploreEnv(maps=('office',), seed=0, pre_explore=1.0)
    pre.reset(seed=2)
    assert pre.known > env.known and pre.trail.sum() >= 2, 'pre-explored start must reveal more map'
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
