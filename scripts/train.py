import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import yaml
import torch

from agents.ppo import PPOAgent
from agents.reinforce import REINFORCEAgent
from envs.partial_obs_cartpole import make_partial_obs_cartpole


def make_env(env_name):
    if env_name == 'CartPole-v1-partial':
        return make_partial_obs_cartpole()
    elif env_name == 'CartPole-v1':
        import gymnasium as gym
        return gym.make('CartPole-v1')
    else:
        raise ValueError(f"Unknown env '{env_name}'. Choose: CartPole-v1, CartPole-v1-partial")


def make_agent(agent_name, agent_params, n_observations, n_actions, device):
    if agent_name == 'ppo':
        return PPOAgent(
            n_observations=n_observations,
            n_actions=n_actions,
            device=device,
            **agent_params,
        )
    elif agent_name == 'reinforce':
        return REINFORCEAgent(
            n_observations=n_observations,
            n_actions=n_actions,
            device=device,
            **agent_params,
        )
    else:
        raise ValueError(f"Unknown agent '{agent_name}'. Choose: ppo, reinforce")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to YAML config file')
    parser.add_argument('--save', required=True, help='Path to save trained model (.pth)')
    parser.add_argument('--n_episodes', type=int, default=1000, help='Number of training episodes')
    parser.add_argument('--no_plot', action='store_true', help='Disable live training plot')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    env = make_env(cfg['env'])
    state, _ = env.reset()
    n_observations = len(state)
    n_actions = env.action_space.n
    print(f"Env: {cfg['env']}  |  obs={n_observations}  actions={n_actions}")

    agent = make_agent(
        agent_name=cfg['agent'],
        agent_params=cfg.get('agent_params', {}),
        n_observations=n_observations,
        n_actions=n_actions,
        device=device,
    )
    print(f"Agent: {cfg['agent']}  |  {cfg.get('agent_params', {})}")

    print(f"Training for {args.n_episodes} episodes...")
    agent.train(env, args.n_episodes, plot_results=not args.no_plot)

    save_path = Path(args.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    agent.save_model(str(save_path))


if __name__ == '__main__':
    main()
