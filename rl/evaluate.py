"""Compare policies in the 2D sim: PPO (via the exported ONNX, same as the ROS node), frontier baseline, random.
usage: python evaluate.py ../models/ppo_explore.onnx [episodes_per_map]"""
import sys
from collections import deque
import numpy as np
import onnxruntime as ort
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from env import ExploreEnv, ACTIONS
from mapping import RES, FREE, OCC, UNKNOWN, dilate

MAX_STEPS = 1000                         # 1000 s of robot time


def ppo_policy(path):
    sess = ort.InferenceSession(path)
    return lambda env, obs: int(np.argmax(sess.run(None, {'obs': obs[None]})[0]))


def random_policy(env, obs):
    return int(env.np_random.integers(len(ACTIONS)))


def frontier_policy(env, obs):
    """Classic baseline: BFS on the known map to the nearest frontier, steer at a point ~0.5 m along the route."""
    b = env.belief
    if env.steps == 0:
        env.bumps, env.last_coll, env.goal, env.stall = np.zeros(b.shape, bool), 0, None, (0, 0)
    if env.known > env.stall[0]:
        env.stall = (env.known, env.steps)
    elif env.steps - env.stall[1] >= 30 and env.goal:   # 30 s without new map: blacklist the goal (as lidar_explorer)
        env.bumps[env.goal] = True
        env.stall = (env.known, env.steps)
    if env.collisions > env.last_coll:                   # bumper: remember the cell we ran into
        env.last_coll = env.collisions
        r, c = env._cell(env.pose + [0.25 * np.cos(env.pose[2]), 0.25 * np.sin(env.pose[2]), 0])
        env.bumps[min(max(r, 0), b.shape[0] - 1), min(max(c, 0), b.shape[1] - 1)] = True
    safe = (b == FREE) & ~dilate(b == OCC, 2) & ~env.bumps
    walk = safe & ~dilate((b == OCC) | env.bumps, 3)      # +0.1 m margin so heading drift doesn't rub walls
    unk = b == UNKNOWN
    start = env._cell(env.pose)
    for trav in (walk, safe):                            # tight spots: retry without the margin
        front = trav & (np.roll(unk, 1, 0) | np.roll(unk, -1, 0) | np.roll(unk, 1, 1) | np.roll(unk, -1, 1))
        prev, q, goal = {start: None}, deque([start]), None
        while q:
            cur = q.popleft()
            if front[cur] and abs(cur[0] - start[0]) + abs(cur[1] - start[1]) > 5:
                goal = cur
                break
            for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                n = (cur[0] + d[0], cur[1] + d[1])
                near = abs(n[0] - start[0]) + abs(n[1] - start[1]) <= 3    # escape zone when inside inflation
                if 0 <= n[0] < b.shape[0] and 0 <= n[1] < b.shape[1] and n not in prev and \
                        (trav[n] or near and safe[n]):
                    prev[n] = cur
                    q.append(n)
        if goal is not None:
            env.goal = goal
            break
    else:
        return 3                                         # nothing reachable: spin
    path = [goal]
    while prev[path[-1]] is not None:
        path.append(prev[path[-1]])
    tgt = path[max(len(path) - 6, 0)]
    err = np.arctan2(tgt[0] - start[0], tgt[1] - start[1]) - env.pose[2]
    err = (err + np.pi) % (2 * np.pi) - np.pi
    if abs(err) > 0.6:
        return 3 if err > 0 else 4
    if abs(err) > 0.15:
        return 1 if err > 0 else 2
    return 0


def run(env, policy, seed):
    obs, _ = env.reset(seed=seed)
    cov, traj = [env.coverage()], [env.pose[:2].copy()]
    for _ in range(MAX_STEPS):
        obs, _, done, _, info = env.step(policy(env, obs))
        cov.append(info['coverage'])
        traj.append(env.pose[:2].copy())
        if done:
            break
    cov += [cov[-1]] * (MAX_STEPS + 1 - len(cov))
    return np.array(cov), np.array(traj), env.collisions


if __name__ == '__main__':
    onnx_path = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    policies = {'ppo': ppo_policy(onnx_path), 'frontier': frontier_policy, 'random': random_policy}
    out = onnx_path.replace('.onnx', '')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    lines = []
    for ax, m in zip(axes, ('office', 'market')):
        env = ExploreEnv(maps=(m,), max_steps=MAX_STEPS)
        for name, pol in policies.items():
            res = [run(env, pol, 1000 + s) for s in range(n)]          # same spawns for every policy
            covs = np.array([r[0] for r in res])
            t90 = [int(np.argmax(c >= 0.9)) if (c >= 0.9).any() else None for c in covs]
            lines.append(f'{m:7s} {name:9s} final cov {covs[:, -1].mean():.3f} ± {covs[:, -1].std():.3f}  '
                         f'cov@300s {covs[:, 300].mean():.3f}  reached 90%: {sum(t is not None for t in t90)}/{n}'
                         f'  t90 median {np.median([t for t in t90 if t is not None]) if any(t90) else "-"} s  '
                         f'collisions {np.mean([r[2] for r in res]):.1f}')
            print(lines[-1], flush=True)
            ax.plot(covs.mean(0), label=name)
            if name == 'ppo':
                cov, traj, _ = run(env, pol, 1000)        # replay episode 0 (deterministic) for its map
                plt.figure(figsize=(10, 6))
                plt.imshow(env.belief, origin='lower', cmap='gray_r', vmin=-1, vmax=1,
                           extent=[env.w.origin[0], env.w.origin[0] + env.belief.shape[1] * RES,
                                   env.w.origin[1], env.w.origin[1] + env.belief.shape[0] * RES])
                plt.plot(traj[:, 0], traj[:, 1], 'r-', lw=1)
                plt.plot(*traj[0], 'go')
                plt.title(f'PPO on {m}: coverage {cov[-1]:.2f} after {MAX_STEPS} s')
                plt.savefig(f'{out}_{m}_episode.png', dpi=90, bbox_inches='tight')
                plt.close()
        ax.set_title(m); ax.set_xlabel('time (s)'); ax.set_ylabel('coverage'); ax.legend(); ax.grid(alpha=.3)
    fig.savefig(f'{out}_coverage.png', dpi=90, bbox_inches='tight')
    open(f'{out}_eval.txt', 'w').write('\n'.join(lines) + '\n')
