"""PPO training + ONNX export.
usage: python train.py STEPS NAME [macro]   train, save ../models/NAME.zip + .onnx
                                            macro: hierarchical env (policy picks frontier candidates)
       python train.py path/to/model.zip    export that model to path/to/model.onnx only"""
import sys
import torch
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from env import ExploreEnv, MacroExploreEnv

OUT = '../models'
N_ENVS = 12


class _Policy(torch.nn.Module):
    """obs -> action logits. extract_features applies SB3's /255 itself, so the ONNX takes raw uint8 maps.
    Flat models take one input `obs`; macro models take `map` and `cands`."""
    def __init__(self, policy):
        super().__init__()
        self.p = policy

    def forward(self, *inputs):
        obs = dict(zip(self.p.observation_space.spaces, inputs)) if len(inputs) > 1 else inputs[0]
        feats = self.p.extract_features(obs, self.p.pi_features_extractor)
        return self.p.action_net(self.p.mlp_extractor.forward_actor(feats))


def export_onnx(model, path):
    model.policy.to('cpu').eval()
    space = model.observation_space
    if isinstance(space, spaces.Dict):
        names = list(space.spaces)
        dummy = tuple(torch.zeros((1, *s.shape), dtype=torch.uint8 if s.dtype == 'uint8' else torch.float32)
                      for s in space.spaces.values())
    else:
        names, dummy = ['obs'], (torch.zeros((1, *space.shape), dtype=torch.uint8),)
    torch.onnx.export(_Policy(model.policy), dummy, path, input_names=names, output_names=['logits'],
                      dynamo=False)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1].endswith('.zip'):
        export_onnx(PPO.load(sys.argv[1], device='cpu'), sys.argv[1][:-4] + '.onnx')
        sys.exit()
    STEPS = int(float(sys.argv[1])) if len(sys.argv) > 1 else 3_000_000
    NAME = sys.argv[2] if len(sys.argv) > 2 else 'ppo_explore'
    macro = len(sys.argv) > 3 and sys.argv[3] == 'macro'
    torch.set_num_threads(4)
    if macro:
        # one step = one frontier choice (~5-10 s of robot time), so far fewer steps and a shorter rollout
        venv = make_vec_env(MacroExploreEnv, n_envs=N_ENVS, env_kwargs={'pre_explore': 0.5},
                            vec_env_cls=SubprocVecEnv, seed=0)
        model = PPO('MultiInputPolicy', venv, n_steps=128, batch_size=256, n_epochs=4, learning_rate=2.5e-4,
                    gamma=0.99, gae_lambda=0.95, clip_range=0.1, ent_coef=0.01, verbose=1, device='cpu', seed=0)
        every = 50_000
    else:
        venv = make_vec_env(ExploreEnv, n_envs=N_ENVS, env_kwargs={'pre_explore': 0.5},
                            vec_env_cls=SubprocVecEnv, seed=0)
        model = PPO('CnnPolicy', venv, n_steps=256, batch_size=512, n_epochs=4, learning_rate=2.5e-4,
                    gamma=0.995, gae_lambda=0.95, clip_range=0.1, ent_coef=0.01, verbose=1, device='cpu', seed=0)
        every = 500_000
    model.learn(STEPS, callback=CheckpointCallback(every // N_ENVS, f'{OUT}/ckpt', NAME), progress_bar=False)
    model.save(f'{OUT}/{NAME}')
    export_onnx(model, f'{OUT}/{NAME}.onnx')
    print('saved', f'{OUT}/{NAME}.zip', f'{OUT}/{NAME}.onnx')
