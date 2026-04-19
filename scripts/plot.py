import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import re
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# Canonical display order (earlier = left/first bar)
CONFIG_ORDER = [
    'ppo_classic',
    'ppo_note',
    'ppo_note_blend',
    'ppo_note_ow',
    'reinforce_classic',
    'reinforce_note',
]

# Display names for config stems
CONFIG_LABELS = {
    'ppo_classic':    'PPO (classic)',
    'ppo_note':       'PPO + Note',
    'ppo_note_blend': 'PPO + Note (blend gate)',
    'ppo_note_ow':    'PPO + Note (overwrite gate)',
    'reinforce_classic': 'REINFORCE (classic)',
    'reinforce_note':    'REINFORCE + Note',
}


def moving_average(data, window):
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode='valid')


def load_groups(models_dir):
    """Discover .pth files and group them by config name (strip _runN suffix)."""
    paths = sorted(Path(models_dir).glob('*.pth'))
    if not paths:
        raise FileNotFoundError(f"No .pth files found in '{models_dir}'")

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

    return dict(sorted(groups.items(), key=lambda kv: order_key(kv[0])))


def load_rewards(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    rewards = checkpoint.get('episode_rewards', [])
    if not rewards:
        print(f"Warning: no episode_rewards in {path}, skipping.")
    return np.array(rewards, dtype=np.float32)


def plot_learning_curves(ax, groups, window, colors):
    for (name, paths), color in zip(groups.items(), colors):
        runs = [load_rewards(p) for p in paths]
        runs = [r for r in runs if len(r) > 0]
        if not runs:
            continue

        min_len = min(len(r) for r in runs)
        if min_len < window:
            print(f"Warning: '{name}' has only {min_len} episodes, less than window={window}. Skipping.")
            continue
        smoothed = np.stack([moving_average(r[:min_len], window) for r in runs])
        mean = smoothed.mean(axis=0)
        std = smoothed.std(axis=0)
        x = np.arange(window - 1, min_len)  # valid range starts at window-1

        label = CONFIG_LABELS.get(name, name)
        ax.plot(x, mean, color=color, linewidth=1.5, label=label)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.2)

    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('Learning Curves')
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f'{int(v):,}'))
    ax.legend(fontsize=8)


def plot_bar_chart(ax, groups, colors, eval_results=None):
    """Bar chart of final performance.

    If eval_results is provided (dict from evaluate.py JSON), use those values.
    Otherwise falls back to mean of last 100 training episodes.
    """
    names, means, stds, bar_colors = [], [], [], []
    for (name, _paths), color in zip(groups.items(), colors):
        if eval_results is not None:
            entry = eval_results.get(name)
            if entry is None:
                print(f"Warning: no eval results for '{name}', skipping bar.")
                continue
            names.append(name)
            means.append(entry['mean'])
            stds.append(entry['std'])
            bar_colors.append(color)
        else:
            paths = _paths
            runs = [load_rewards(p) for p in paths]
            runs = [r for r in runs if len(r) > 0]
            if not runs:
                continue
            final = np.array([r[-100:].mean() for r in runs])
            names.append(name)
            means.append(float(final.mean()))
            stds.append(float(final.std()))
            bar_colors.append(color)

    x = np.arange(len(names))
    ax.bar(x, means, yerr=stds, color=bar_colors, alpha=0.85,
           capsize=4, error_kw={'linewidth': 1.2})
    ax.set_xticks(x)
    ax.set_xticklabels(
        [CONFIG_LABELS.get(n, n) for n in names],
        rotation=20, ha='right', fontsize=8,
    )
    ylabel = 'Mean Eval Reward' if eval_results is not None else 'Mean Reward (last 100 train ep.)'
    ax.set_ylabel(ylabel)
    ax.set_title('Final Performance')


def main():
    plt.rcParams.update({
        'font.size': 10,
        'lines.linewidth': 1.5,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })

    parser = argparse.ArgumentParser()
    parser.add_argument('--models_dir', required=True, help='Directory containing .pth model files')
    parser.add_argument('--window', type=int, default=100, help='Moving average window (default: 100)')
    parser.add_argument('--eval_results', default=None, help='Path to JSON from evaluate.py for bar chart')
    parser.add_argument('--save', default=None, help='Save figure to this path (e.g. plots/cartpole.png)')
    parser.add_argument('--show', action='store_true', help='Show figure interactively')
    args = parser.parse_args()

    eval_results = None
    if args.eval_results:
        with open(args.eval_results) as f:
            eval_results = json.load(f)

    groups = load_groups(args.models_dir)
    env_name = Path(args.models_dir).name
    n = len(groups)
    colors = plt.colormaps['tab10'](np.linspace(0, 0.9, n))

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(env_name, fontsize=12, fontweight='bold')

    plot_learning_curves(ax_left, groups, args.window, colors)
    plot_bar_chart(ax_right, groups, colors, eval_results=eval_results)

    plt.tight_layout()

    if args.save:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    if args.show or args.save is None:
        plt.show()


if __name__ == '__main__':
    main()