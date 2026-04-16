import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt


def moving_average(data, window):
    result = np.zeros(len(data))
    for i in range(window, len(data)):
        result[i] = np.mean(data[i - window:i])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('models', nargs='+', help='One or more .pth checkpoint paths')
    parser.add_argument('--window', type=int, default=100, help='Moving average window size (default: 100)')
    args = parser.parse_args()

    plt.figure(figsize=(12, 6))

    for path in args.models:
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        durations = checkpoint.get('episode_durations', [])
        if not durations:
            print(f"Warning: no episode_durations in {path}, skipping.")
            continue

        label = Path(path).stem
        data = np.array(durations, dtype=np.float32)
        ma = moving_average(data, args.window)

        plt.plot(data, alpha=0.3)
        line, = plt.plot(ma, linewidth=2, label=label)

    plt.xlabel('Episode')
    plt.ylabel('Duration (steps)')
    plt.title('Training Episode Durations')
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()