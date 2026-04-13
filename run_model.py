import gymnasium as gym
import torch
import numpy as np

from partial_obs_cartpole import make_partial_obs_cartpole
from reinforce import REINFORCEAgent
from ppo import PPOAgent

import matplotlib.pyplot as plt

def run_trained_model(env, agent, num_episodes=5):
    
    # Set to evaluation mode
    agent.eval()
    
    print(f"Running {num_episodes} episodes with trained agent...\n")
    
    episode_rewards = []
    
    # Run episodes
    for episode in range(num_episodes):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)
        if agent.n_notes is not None:
            agent.note_array = np.zeros((agent.n_notes), dtype=np.float32)
        
        total_reward = 0
        steps = 0
        
        while True:
            action = agent.select_action(state, deterministic=True)
            
            # Take action in environment
            observation, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1
            
            done = terminated or truncated
            
            if not terminated:
                state = torch.tensor(observation, dtype=torch.float32, device=agent.device).unsqueeze(0)
            
            if done:
                episode_rewards.append(total_reward)
                #print(f"Episode {episode + 1}: Steps = {steps}, Total Reward = {total_reward}")
                break
    
    # Print statistics
    print(f"\n{'='*50}")
    print(f"Average reward over {num_episodes} episodes: {sum(episode_rewards) / len(episode_rewards):.2f}")
    print(f"Best episode: {max(episode_rewards):.0f} steps")
    print(f"Worst episode: {min(episode_rewards):.0f} steps")
    print(f"{'='*50}")

def moving_average(data, window_size=1000):
    result = np.zeros(len(data))
    for i in range(window_size, len(data)):
        result[i] = np.mean(data[i-window_size:i])
    return result

if __name__ == "__main__":
    render_mode = "human"
    #render_mode = None

    #env = gym.make("CartPole-v1", render_mode=render_mode)
    env = make_partial_obs_cartpole(render_mode=render_mode)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get environment dimensions
    n_actions = env.action_space.n
    state, info = env.reset()
    n_observations = len(state)

    ppo_model_classic_path = 'models\\ppo_classic.pth'
    ppo_model_note_path = 'models\\ppo_note.pth'
    ppo_model_gated_path = 'models\\ppo_gated.pth'

    '''
    note_agent = REINFORCEAgent(
        n_observations=n_observations,
        n_actions=n_actions,
        n_notes=4,
        device=device
    )
    note_agent.load_model(note_model_path)

    classic_agent = REINFORCEAgent(
        n_observations=n_observations,
        n_actions=n_actions,
        n_notes=None,
        device=device
    )
    classic_agent.load_model(classic_model_path)
'''
    
    ppo_agent_classic = PPOAgent(
        n_observations=n_observations,
        n_actions=n_actions,
        n_notes=None,
        write_gate=False,
        hidden_size=64,
        device=device
    )
    ppo_agent_classic.load_model(ppo_model_classic_path)

    ppo_agent_note = PPOAgent(
        n_observations=n_observations,
        n_actions=n_actions,
        n_notes=4,
        write_gate=False,
        hidden_size=64,
        device=device
    )
    ppo_agent_note.load_model(ppo_model_note_path)

    ppo_agent_gated = PPOAgent(
        n_observations=n_observations,
        n_actions=n_actions,
        n_notes=4,
        write_gate=True,
        hidden_size=64,
        device=device
    )
    ppo_agent_gated.load_model(ppo_model_gated_path)

    #run_trained_model(env, agent=ppo_agent_classic, num_episodes=100)
    run_trained_model(env, agent=ppo_agent_note, num_episodes=10)
    #run_trained_model(env, agent=ppo_agent_gated, num_episodes=10)

    '''
    train_episode_rewards_ppo_classic = ppo_agent_classic.episode_durations
    train_episode_rewards_ppo_note = ppo_agent_note.episode_durations
    train_episode_rewards_ppo_gated = ppo_agent_gated.episode_durations

    avg_rewards_ppo_classic = moving_average(train_episode_rewards_ppo_classic, window_size=1000)
    avg_rewards_ppo_note = moving_average(train_episode_rewards_ppo_note, window_size=1000)
    avg_rewards_ppo_gated = moving_average(train_episode_rewards_ppo_gated, window_size=1000)

    plt.figure(figsize=(12, 6))
    plt.plot(train_episode_rewards_ppo_classic, color='lightgreen', alpha=0.3, label='PPO Classic')
    plt.plot(train_episode_rewards_ppo_note, color='coral', alpha=0.3, label='PPO + Note')
    plt.plot(train_episode_rewards_ppo_gated, color='seagreen', alpha=0.3, label='PPO + Gated Note')
    plt.plot(range(len(avg_rewards_ppo_classic)), avg_rewards_ppo_classic, color='darkgreen', label='PPO Classic (MA)', linewidth=2)
    plt.plot(range(len(avg_rewards_ppo_note)), avg_rewards_ppo_note, color='darkred', label='PPO + Note (MA)', linewidth=2)
    plt.plot(range(len(avg_rewards_ppo_gated)), avg_rewards_ppo_gated, color='darkblue', label='PPO + Gated Note (MA)', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Duration (Steps)')
    plt.title('Training Episode Durations')
    plt.legend()
    plt.show()
'''