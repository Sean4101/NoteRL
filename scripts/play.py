import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import numpy as np
import torch

from agents.ppo import PPOAgent
from agents.reinforce import REINFORCEAgent
from envs.partial_obs_cartpole import make_partial_obs_cartpole


AGENT_CLASSES = {
    'ppo': PPOAgent,
    'reinforce': REINFORCEAgent,
}


def make_env(env_name, render_mode):
    if env_name == 'CartPole-v1-partial':
        return make_partial_obs_cartpole(render_mode=render_mode)
    elif env_name == 'CartPole-v1':
        import gymnasium as gym
        return gym.make('CartPole-v1', render_mode=render_mode)
    else:
        raise ValueError(f"Unknown env '{env_name}'.")


def load_agent(model_path, device):
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if 'config' not in checkpoint:
        raise ValueError(f"Checkpoint '{model_path}' has no config.")
    cls = AGENT_CLASSES.get(checkpoint.get('agent'))
    if cls is None:
        raise ValueError(f"Unknown or missing agent type in checkpoint: '{checkpoint.get('agent')}'")
    return cls.from_checkpoint(model_path, device=device)


def run(env, agent, num_episodes):
    agent.eval()
    print(f"Running {num_episodes} episodes...\n")

    episode_rewards = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)
        if agent.n_notes is not None:
            agent.note_array = np.zeros(agent.n_notes, dtype=np.float32)

        total_reward = 0

        while True:
            action = agent.select_action(state, deterministic=True)
            observation, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated

            if not terminated:
                state = torch.tensor(observation, dtype=torch.float32, device=agent.device).unsqueeze(0)

            if done:
                episode_rewards.append(total_reward)
                print(f"Episode {episode + 1:>3}: {total_reward:.0f} steps")
                break

    print(f"\n{'='*40}")
    print(f"Episodes   : {num_episodes}")
    print(f"Average    : {sum(episode_rewards) / len(episode_rewards):.2f}")
    print(f"Best       : {max(episode_rewards):.0f}")
    print(f"Worst      : {min(episode_rewards):.0f}")
    print(f"{'='*40}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Path to trained model checkpoint (.pth)')
    parser.add_argument('--env', required=True, help='Environment name: CartPole-v1 | CartPole-v1-partial')
    parser.add_argument('--n_episodes', type=int, default=10, help='Number of episodes to run (default: 10)')
    parser.add_argument('--no_render', action='store_true', help='Disable rendering')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    render_mode = None if args.no_render else 'human'
    env = make_env(args.env, render_mode=render_mode)

    agent = load_agent(args.model, device=device)
    print(f"Loaded: {args.model}  |  {type(agent).__name__}")

    run(env, agent, args.n_episodes)
    env.close()


if __name__ == '__main__':
    main()
