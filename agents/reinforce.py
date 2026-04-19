from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


class PolicyNetwork(nn.Module):
    def __init__(self, n_observations, n_actions, n_notes=None, hidden_size=64):
        super(PolicyNetwork, self).__init__()
        input_size = n_observations if n_notes is None else n_observations + n_notes
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, n_actions)
        if n_notes is not None:
            self.fc_note = nn.Linear(hidden_size, n_notes)
        self.n_notes = n_notes
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action_probs = F.softmax(self.fc3(x), dim=-1)
        if self.n_notes is None:
            return action_probs
        return action_probs, self.fc_note(x)

class REINFORCEAgent:
    def __init__(self, n_observations, n_actions, n_notes=None, hidden_size=64, lr=1e-2, gamma=0.99, device=None):
        self.n_observations = n_observations
        self.n_actions = n_actions
        self.n_notes = n_notes
        self.gamma = gamma
        self.device = device if device else torch.device("cpu")
        self._config = {
            'n_observations': n_observations,
            'n_actions': n_actions,
            'n_notes': n_notes,
            'hidden_size': hidden_size,
            'lr': lr,
            'gamma': gamma,
        }
        
        self.policy_network = PolicyNetwork(
            n_observations,
            n_actions,
            n_notes,
            hidden_size
        ).to(self.device)

        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=lr)

        if n_notes is not None:
            self.note_array = np.zeros((n_notes), dtype=np.float32)
        
        # Training tracking
        self.episode_rewards = []
    
    def act(self, state):
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)

        if self.n_notes is None:
            action_probs = self.policy_network(state_tensor)
            m = Categorical(action_probs)
            action = m.sample()
            return action.item(), m.log_prob(action)
        else:
            state_with_notes = torch.cat((state_tensor, torch.from_numpy(self.note_array).float().unsqueeze(0).to(self.device)), dim=1)
            action_probs, note_values = self.policy_network(state_with_notes)
            m = Categorical(action_probs)
            action = m.sample()
            self.note_array = note_values.detach().cpu().numpy().flatten()
            return action.item(), m.log_prob(action)
    
    def train(self, env, n_training_episodes, max_t=1000, plot_results=True):
        if plot_results:
            plt.ion()
        for i_episode in range(n_training_episodes):
            saved_log_probs = []
            rewards = []
            state, _ = env.reset()
            if self.n_notes is not None:
                self.note_array = np.zeros((self.n_notes), dtype=np.float32)
            
            for t in range(max_t):
                action, log_prob = self.act(state)
                saved_log_probs.append(log_prob)
                state, reward, terminated, truncated, _ = env.step(action)
                rewards.append(reward)
                done = terminated or truncated
                if done:
                    break

            returns = deque(maxlen=max_t)
            n_steps = len(rewards)

            for t_return in range(n_steps)[::-1]:
                disc_return_t = returns[0] if len(returns) > 0 else 0
                returns.appendleft(self.gamma * disc_return_t + rewards[t_return])

            eps = np.finfo(np.float32).eps.item()
            
            returns = torch.tensor(returns)
            returns = (returns - returns.mean()) / (returns.std() + eps)

            policy_loss = []
            for log_prob, disc_return in zip(saved_log_probs, returns):
                policy_loss.append(-log_prob * disc_return)
            policy_loss = torch.cat(policy_loss).sum()

            self.optimizer.zero_grad()
            policy_loss.backward()
            self.optimizer.step()
            
            self.episode_rewards.append(sum(rewards))
            if (i_episode + 1) % 100 == 0:
                avg = np.mean(self.episode_rewards[-100:])
                print(f"Episode {i_episode + 1}/{n_training_episodes}  |  Avg reward (last 100): {avg:.1f}")
            if plot_results:
                self.plot_rewards()

        print('Training Complete')
        if plot_results:
            self.plot_rewards(show_result=True)
            plt.ioff()
            plt.show()
    
    def plot_rewards(self, show_result=False):
        plt.figure(1)
        rewards_t = torch.tensor(self.episode_rewards, dtype=torch.float)
        
        if show_result:
            plt.title('Result')
        else:
            plt.clf()
            plt.title('Training...')
        
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.plot(rewards_t.numpy(), alpha=0.3)
        
        if len(rewards_t) >= 1000:
            means = rewards_t.unfold(0, 1000, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(999), means))
            plt.plot(means.numpy())

        plt.pause(0.001)
    
    def eval(self):
        """Set network to evaluation mode."""
        self.policy_network.eval()
    
    def train_mode(self):
        """Set network to training mode."""
        self.policy_network.train()
    
    def select_action(self, state, deterministic=False):
        with torch.no_grad():
            if self.n_notes is not None:
                state_with_notes = torch.cat((state, torch.from_numpy(self.note_array).float().unsqueeze(0).to(self.device)), dim=1)
                probs, _ = self.policy_network(state_with_notes)
            else:
                probs = self.policy_network(state)
            
            if deterministic:
                return probs.argmax().item()
            else:
                m = Categorical(probs)
                action = m.sample()
                return action.item()
    
    def save_model(self, filepath):
        torch.save({
            'agent': 'reinforce',
            'config': self._config,
            'policy_net_state_dict': self.policy_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episode_rewards': self.episode_rewards,
        }, filepath)
        print(f'Model saved to: {filepath}')

    # Deprecated: Use from_checkpoint() instead for loading with config
    def load_model(self, filepath):
        checkpoint = torch.load(filepath, weights_only=False)
        self.policy_network.load_state_dict(checkpoint['policy_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.episode_rewards = checkpoint['episode_rewards']
        print(f'Model loaded from: {filepath}')

    @classmethod
    def from_checkpoint(cls, filepath, device=None):
        checkpoint = torch.load(filepath, map_location=device, weights_only=False)
        if 'config' not in checkpoint:
            raise ValueError(f"Checkpoint '{filepath}' has no config. Use load_model() with a manually constructed agent instead.")
        agent = cls(**checkpoint['config'], device=device)
        agent.policy_network.load_state_dict(checkpoint['policy_net_state_dict'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        agent.episode_rewards = checkpoint['episode_rewards']
        print(f'Model loaded from: {filepath}')
        return agent
