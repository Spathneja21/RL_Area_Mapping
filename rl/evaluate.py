"""Compare exploration policies in the 2D sim on the same spawns.
usage: python evaluate.py MODEL.onnx [episodes_per_map]

ppo      the ONNX model, actions sampled (flat 6/8-channel models on ExploreEnv, macro models on MacroExploreEnv)
greedy   always the nearest frontier (slot 0) with the shared BFS planner: the classic frontier explorer
random   a random frontier candidate each decision, same planner
"""
import sys
import numpy as np
import onnxruntime as ort
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from env import ExploreEnv, MacroExploreEnv
from mapping import RES

MAX_STEPS = 1000                         # 1000 s of robot time


def _sample(env, logits):
    p = np.exp(logits - logits.max())
    return int(env.np_random.choice(len(p), p=p / p.sum()))


def ppo_policy(path):
    """-> (env class, act). Samples from the policy, as in training (argmax pins the flat policy to walls)."""
    sess = ort.InferenceSession(path)
    inputs = [i.name for i in sess.get_inputs()]
    if len(inputs) > 1:                                    # macro: inputs 'cands' and 'map'
        return MacroExploreEnv, lambda env, obs: _sample(env, sess.run(None, {k: obs[k][None] for k in inputs})[0][0])
    chans = [0, 1, 2, 4, 5, 6] if sess.get_inputs()[0].shape[1] == 6 else slice(None)   # v1: no frontier channel
    return ExploreEnv, lambda env, obs: _sample(env, sess.run(None, {'obs': obs[None, chans]})[0][0])


def greedy_policy(env, obs):
    return 0


def random_frontier_policy(env, obs):
    return int(env.np_random.integers(max(len(env.goals), 1)))


def run(env, policy, seed):
    """One episode -> (coverage per second, padded to MAX_STEPS + 1; trajectory; collisions)."""
    obs, _ = env.reset(seed=seed)
    done = trunc = False
    while not (done or trunc):
        obs, _, done, trunc, _ = env.step(policy(env, obs))
    cov = env.cov_log[:MAX_STEPS + 1]
    return np.array(cov + [cov[-1]] * (MAX_STEPS + 1 - len(cov))), np.array(env.traj), env.collisions


if __name__ == '__main__':
    onnx_path = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    ppo_env, ppo = ppo_policy(onnx_path)
    policies = {'ppo': (ppo_env, ppo), 'greedy': (MacroExploreEnv, greedy_policy),
                'random': (MacroExploreEnv, random_frontier_policy)}
    out = onnx_path.replace('.onnx', '')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    lines = []
    for ax, m in zip(axes, ('office', 'market')):
        for name, (env_cls, pol) in policies.items():
            env = env_cls(maps=(m,), max_steps=MAX_STEPS)
            res = [run(env, pol, 1000 + s) for s in range(n)]          # same spawns for every policy
            covs = np.array([r[0] for r in res])
            t90 = [int(np.argmax(c >= 0.9)) for c in covs if (c >= 0.9).any()]
            lines.append(f'{m:7s} {name:7s} final cov {covs[:, -1].mean():.3f} ± {covs[:, -1].std():.3f}  '
                         f'cov@300s {covs[:, 300].mean():.3f}  reached 90%: {len(t90)}/{n}  '
                         f't90 median {np.median(t90) if t90 else "-"} s  '
                         f'collisions {np.mean([r[2] for r in res]):.1f}')
            print(lines[-1], flush=True)
            ax.plot(covs.mean(0), label=name)
            if name == 'ppo':
                cov, traj, _ = res[0]
                run(env, pol, 1000)                       # replay episode 0 (seeded, deterministic) for its map
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
