import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import re
import numpy as np
import torch

from agents.ppo import PPOAgent
from agents.reinforce import REINFORCEAgent
from envs.partial_obs_cartpole import make_partial_obs_cartpole
from envs.minigrid_flat import make_minigrid_flat


AGENT_CLASSES = {
    'ppo': PPOAgent,
    'reinforce': REINFORCEAgent,
}

# Mapping from models/ sub-directory names to gym env ids
ENV_MAP = {
    'cartpole-partial': 'CartPole-v1-partial',
    'minigrid-memory':  'MiniGrid-MemoryS7-v0',
}

# Canonical display order — must match plot.py
CONFIG_ORDER = [
    'ppo_classic',
    'ppo_note',
    'ppo_note_blend',
    'ppo_note_ow',
    'reinforce_classic',
    'reinforce_note',
]


def make_env(env_name):
    if env_name == 'CartPole-v1-partial':
        return make_partial_obs_cartpole()
    return make_minigrid_flat(env_name)


def load_agent(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    cls = AGENT_CLASSES.get(checkpoint.get('agent'))
    if cls is None:
        raise ValueError(f"Unknown agent type in '{path}': {checkpoint.get('agent')}")
    return cls.from_checkpoint(str(path), device=device)


def evaluate_agent(agent, env, n_episodes):
    agent.eval()
    rewards = []
    for _ in range(n_episodes):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)
        if agent.n_notes is not None:
            agent.note_array = np.zeros(agent.n_notes, dtype=np.float32)
        total = 0.0
        while True:
            action = agent.select_action(state, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            if terminated or truncated:
                break
            state = torch.tensor(obs, dtype=torch.float32, device=agent.device).unsqueeze(0)
        rewards.append(total)
    return rewards


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models_dir', required=True,
                        help='Directory of .pth files (e.g. models/cartpole-partial)')
    parser.add_argument('--env', default=None,
                        help='Gym env id. Auto-detected from directory name if omitted.')
    parser.add_argument('--n_episodes', type=int, default=50,
                        help='Evaluation episodes per model run (default: 50)')
    parser.add_argument('--save', default=None,
                        help='Save results JSON to this path')
    parser.add_argument('--save_txt', default=None,
                        help='Save results as a human-readable text report to this path')
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    env_name = args.env or ENV_MAP.get(models_dir.name)
    if env_name is None:
        raise ValueError(
            f"Cannot infer env from directory '{models_dir.name}'. "
            f"Pass --env explicitly. Known directories: {list(ENV_MAP)}"
        )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Group .pth files by config name
    paths = sorted(models_dir.glob('*.pth'))
    if not paths:
        raise FileNotFoundError(f"No .pth files in '{models_dir}'")

    groups: dict[str, list[Path]] = {}
    for p in paths:
        key = re.sub(r'_run\d+$', '', p.stem)
        groups.setdefault(key, []).append(p)

    # Sort by canonical order; unknown keys go at the end alphabetically
    def order_key(name):
        try:
            return CONFIG_ORDER.index(name)
        except ValueError:
            return len(CONFIG_ORDER)

    groups = dict(sorted(groups.items(), key=lambda kv: order_key(kv[0])))

    print(f"Env         : {env_name}")
    print(f"Episodes    : {args.n_episodes} per run")
    print(f"Configs     : {list(groups)}\n")

    results = {}
    for config_name, run_paths in groups.items():
        env = make_env(env_name)
        run_means = []
        for rp in run_paths:
            agent = load_agent(rp, device)
            ep_rewards = evaluate_agent(agent, env, args.n_episodes)
            run_mean = float(np.mean(ep_rewards))
            run_means.append(run_mean)
            print(f"  {rp.stem}: mean={run_mean:.2f}")
        env.close()

        config_mean = float(np.mean(run_means))
        config_std  = float(np.std(run_means))
        results[config_name] = {'mean': config_mean, 'std': config_std, 'runs': run_means}
        print(f"  → {config_name}: {config_mean:.2f} ± {config_std:.2f}\n")

    if args.save:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved JSON : {save_path}")

    # ── Text report ──────────────────────────────────────────────────────────
    col_w = max(len(k) for k in results) if results else 10
    header = f"{'Config':<{col_w}}  {'Mean':>8}  {'Std':>8}  {'Min run':>8}  {'Max run':>8}  Runs"
    sep = '-' * len(header)
    lines = [
        f"Evaluation Report",
        f"Env        : {env_name}",
        f"Episodes   : {args.n_episodes} per run",
        sep,
        header,
        sep,
    ]
    for config_name, entry in results.items():
        runs_str = '  '.join(f"{r:.2f}" for r in entry['runs'])
        lines.append(
            f"{config_name:<{col_w}}  {entry['mean']:>8.2f}  {entry['std']:>8.2f}"
            f"  {min(entry['runs']):>8.2f}  {max(entry['runs']):>8.2f}  [{runs_str}]"
        )
    lines.append(sep)
    report = '\n'.join(lines) + '\n'

    print()
    print(report)

    if args.save_txt:
        txt_path = Path(args.save_txt)
        txt_path.parent.mkdir(parents=True, exist_ok=True)
        txt_path.write_text(report)
        print(f"Saved text : {txt_path}")


if __name__ == '__main__':
    main()
