import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def moving_average(data, window):
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode='valid')


def main():
    plt.rcParams.update({
        'font.size': 10,
        'lines.linewidth': 1.5,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })

    parser = argparse.ArgumentParser()
    parser.add_argument('models', nargs='+', help='One or more .pth checkpoint paths')
    parser.add_argument('--window', type=int, default=100, help='Moving average window (default: 100)')
    parser.add_argument('--save', default=None, help='Save figure to this path')
    parser.add_argument('--show', action='store_true', help='Show figure interactively')
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(10, 5))

    for path in args.models:
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        rewards = checkpoint.get('episode_rewards', [])
        if not rewards:
            print(f"Warning: no episode_rewards in {path}, skipping.")
            continue

        data = np.array(rewards, dtype=np.float32)
        label = Path(path).stem

        ax.plot(data, alpha=0.25, linewidth=1)
        if len(data) >= args.window:
            ma = moving_average(data, args.window)
            x = np.arange(args.window - 1, len(data))
            line, = ax.plot(x, ma, linewidth=1.8, label=label)
        else:
            # Not enough data for MA — just label the raw line
            ax.lines[-1].set_label(label)
            print(f"Warning: '{label}' has fewer than {args.window} episodes; showing raw only.")

    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('Training Curves')
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f'{int(v):,}'))
    ax.legend(fontsize=8)
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
