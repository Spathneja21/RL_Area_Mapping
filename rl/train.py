"""PPO training + ONNX export.
usage: python train.py [total_steps] [run_name]     train, save ../models/NAME.zip + .onnx
       python train.py path/to/model.zip            export that model to path/to/model.onnx only"""
import sys
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from env import ExploreEnv

OUT = '../models'


class _Policy(torch.nn.Module):
    """uint8 obs -> action logits. extract_features applies SB3's /255 itself, so the ONNX takes raw obs."""
    def __init__(self, policy):
        super().__init__()
        self.p = policy

    def forward(self, obs):
        feats = self.p.extract_features(obs, self.p.pi_features_extractor)
        return self.p.action_net(self.p.mlp_extractor.forward_actor(feats))


def export_onnx(model, path):
    model.policy.to('cpu').eval()
    dummy = torch.zeros((1, *model.observation_space.shape), dtype=torch.uint8)
    torch.onnx.export(_Policy(model.policy), dummy, path, input_names=['obs'], output_names=['logits'],
                      dynamo=False)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1].endswith('.zip'):
        export_onnx(PPO.load(sys.argv[1], device='cpu'), sys.argv[1][:-4] + '.onnx')
        sys.exit()
    STEPS = int(float(sys.argv[1])) if len(sys.argv) > 1 else 3_000_000
    NAME = sys.argv[2] if len(sys.argv) > 2 else 'ppo_explore'
    torch.set_num_threads(4)
    venv = make_vec_env(ExploreEnv, n_envs=12, vec_env_cls=SubprocVecEnv, seed=0)
    model = PPO('CnnPolicy', venv, n_steps=256, batch_size=512, n_epochs=4, learning_rate=2.5e-4,
                gamma=0.99, gae_lambda=0.95, clip_range=0.1, ent_coef=0.01, verbose=1,
                tensorboard_log=None, device='cpu', seed=0)
    model.learn(STEPS, callback=CheckpointCallback(500_000 // 12, f'{OUT}/ckpt', NAME), progress_bar=False)
    model.save(f'{OUT}/{NAME}')
    export_onnx(model, f'{OUT}/{NAME}.onnx')
    print('saved', f'{OUT}/{NAME}.zip', f'{OUT}/{NAME}.onnx')
