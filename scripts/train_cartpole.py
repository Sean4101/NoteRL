import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gymnasium as gym
import torch
import time
from agents.reinforce import REINFORCEAgent
from agents.ppo import PPOAgent

from envs.partial_obs_cartpole import make_partial_obs_cartpole


#env = gym.make("CartPole-v1")
env = make_partial_obs_cartpole()

n_actions = env.action_space.n
state, info = env.reset()
n_observations = len(state)

device = torch.device(
    # "cuda" if torch.cuda.is_available() else
    "cpu"
)
print(f"Using device: {device}")
'''
agent = REINFORCEAgent(
    n_observations=n_observations,
    n_actions=n_actions,
    #n_notes=4,
    hidden_size=64,
    lr=3e-3,
    gamma=0.99,
    device=device
)'''

agent = PPOAgent(
    n_observations=n_observations,
    n_actions=n_actions,
    n_notes=None,
    write_gate=False,
    hidden_size=64,
    lr=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_epsilon=0.1,
    n_epochs=10,
    batch_size=64,
    N=1024,
    device=device
)

num_episodes = 10000

print(f"Training for {num_episodes} episodes...")
agent.train(env, num_episodes, plot_results=True)

save_dir = "models"
timestamp = time.strftime("%Y%m%d-%H%M%S")
agent_name = type(agent).__name__.lower().replace('agent', '')

while True:
    save_model = input("Save model? (y/n): ")
    if save_model.lower() == 'y':
        model_path = f"{save_dir}/{agent_name}_{timestamp}.pth"
        agent.save_model(model_path)
        break
    elif save_model.lower() == 'n':
        print("Model not saved.")
        break
    else:
        print("Invalid input. Please enter 'y' or 'n'.")