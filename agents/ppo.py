from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


class MemoryBuffer:
    def __init__(self, batch_size):
        self.observations = []
        self.log_probs = []
        self.actions = []
        self.values = []
        self.rewards = []
        self.dones = []
        self.batch_size = batch_size

    def generate_batches(self):
        n_observations = len(self.observations)
        batch_start = np.arange(0, n_observations, self.batch_size)
        indices = np.arange(n_observations, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i + self.batch_size] for i in batch_start]

        return (np.array(self.observations),
                np.array(self.actions),
                np.array(self.log_probs),
                np.array(self.values),
                np.array(self.rewards),
                np.array(self.dones),
                batches)

    def store_memory(self, observation, log_prob, action, value, reward, done):
        self.observations.append(observation)
        self.log_probs.append(log_prob)
        self.actions.append(action)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.observations = []
        self.log_probs = []
        self.actions = []
        self.values = []
        self.rewards = []
        self.dones = []

class ActorNetwork(nn.Module):
    def __init__(self, n_observations, n_actions, n_notes=None, hidden_size=256, write_gate=None):
        super(ActorNetwork, self).__init__()
        input_size = n_observations if n_notes is None else n_observations + n_notes
        self.n_notes = n_notes
        self.write_gate = write_gate if n_notes is not None else None

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, n_actions)
        if n_notes is not None:
            self.fc_note = nn.Linear(hidden_size, n_notes)
        if self.write_gate is not None:
            self.fc_gate = nn.Linear(hidden_size, n_notes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        action_probs = torch.softmax(self.fc3(x), dim=-1)
        if self.n_notes is None:
            return Categorical(action_probs)
        note_values = self.fc_note(x)
        if self.write_gate is not None:
            gate_probs = torch.sigmoid(self.fc_gate(x))
            return Categorical(action_probs), note_values, gate_probs
        return Categorical(action_probs), note_values
        
class CriticNetwork(nn.Module):
    def __init__(self, n_observations, n_notes=None, hidden_size=256):
        super(CriticNetwork, self).__init__()
        input_size = n_observations if n_notes is None else n_observations + n_notes
        self.critic = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, observation):
        return self.critic(observation)

class PPOAgent:
    def __init__(self, n_observations, n_actions, n_notes=None, hidden_size=256, lr=3e-4, gamma=0.99,
                 gae_lambda=0.95, clip_epsilon=0.2, n_epochs=10, batch_size=64, N=2048,
                 write_gate=None, entropy_coef=0.01, device=None):
        self.n_notes = n_notes
        self.write_gate = write_gate if n_notes is not None else None
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.n_epochs = n_epochs
        self.N = N
        self.entropy_coef = entropy_coef
        self.device = device if device else torch.device('cpu')
        self._config = {
            'n_observations': n_observations,
            'n_actions': n_actions,
            'n_notes': n_notes,
            'hidden_size': hidden_size,
            'lr': lr,
            'gamma': gamma,
            'gae_lambda': gae_lambda,
            'clip_epsilon': clip_epsilon,
            'n_epochs': n_epochs,
            'batch_size': batch_size,
            'N': N,
            'write_gate': write_gate,
            'entropy_coef': entropy_coef,
        }

        self.actor = ActorNetwork(n_observations, n_actions, n_notes=n_notes, hidden_size=hidden_size, write_gate=write_gate).to(self.device)
        self.critic = CriticNetwork(n_observations, n_notes=n_notes, hidden_size=hidden_size).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        if n_notes is not None:
            self.note_array = np.zeros((n_notes), dtype=np.float32)

        self.memory = MemoryBuffer(batch_size)
        self.episode_rewards = []

    def _apply_note_update(self, new_note, gate_probs=None):
        new_note_np = new_note.detach().cpu().numpy().flatten()
        if gate_probs is None:
            self.note_array = new_note_np
        elif self.write_gate == 'overwrite':
            gate_np = gate_probs.detach().cpu().numpy().flatten()
            mask = np.random.rand(self.n_notes) < gate_np
            self.note_array = np.where(mask, new_note_np, self.note_array)
        elif self.write_gate == 'blend':
            gate_np = gate_probs.detach().cpu().numpy().flatten()
            self.note_array = gate_np * new_note_np + (1 - gate_np) * self.note_array

    def choose_action(self, observation):
        obs = torch.tensor(observation, dtype=torch.float).to(self.device)
        if self.n_notes is None:
            dist = self.actor(obs)
            value = self.critic(obs)
            obs_input_np = observation
        else:
            obs_with_notes = torch.cat((obs, torch.from_numpy(self.note_array).float().to(self.device)), dim=0)
            actor_out = self.actor(obs_with_notes)
            dist, new_note = actor_out[0], actor_out[1]
            gate_probs = actor_out[2] if self.write_gate is not None else None
            self._apply_note_update(new_note, gate_probs)
            value = self.critic(obs_with_notes)
            obs_input_np = obs_with_notes.cpu().numpy()
        action = dist.sample()

        log_prob = torch.squeeze(dist.log_prob(action)).item()
        action = torch.squeeze(action).item()
        value = torch.squeeze(value).item()

        return action, log_prob, value, obs_input_np

    def _learn(self):
        for _ in range(self.n_epochs):
            observations, actions, old_log_probs, values, rewards, dones, batches = \
                self.memory.generate_batches()

            values = np.append(values, 0)
            advantages = np.zeros(len(rewards), dtype=np.float32)

            for t in range(len(rewards) - 1):
                discount = 1
                a_t = 0
                for k in range(t, len(rewards) - 1):
                    a_t += discount * (rewards[k] + self.gamma * values[k + 1] * (1 - int(dones[k])) - values[k])
                    discount *= self.gamma * self.gae_lambda
                advantages[t] = a_t

            advantages = torch.tensor(advantages).to(self.device)
            values_t = torch.tensor(values[:-1]).to(self.device)

            for batch in batches:
                obs = torch.tensor(observations[batch], dtype=torch.float).to(self.device)
                batch_log_probs = torch.tensor(old_log_probs[batch]).to(self.device)
                acts = torch.tensor(actions[batch]).to(self.device)

                actor_out = self.actor(obs)
                dist = actor_out if self.n_notes is None else actor_out[0]
                critic_value = torch.squeeze(self.critic(obs))

                new_log_probs = dist.log_prob(acts)
                prob_ratio = new_log_probs.exp() / batch_log_probs.exp()
                weighted_probs = advantages[batch] * prob_ratio
                weighted_clipped_probs = torch.clamp(prob_ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages[batch]
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantages[batch] + values_t[batch]
                critic_loss = ((returns - critic_value) ** 2).mean()

                entropy_loss = -self.entropy_coef * dist.entropy().mean()
                total_loss = actor_loss + 0.5 * critic_loss + entropy_loss

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()
                self.actor_optimizer.step()
                self.critic_optimizer.step()

        self.memory.clear_memory()

    def train(self, env, n_training_episodes, max_t=1000, plot_results=True):
        if plot_results:
            plt.ion()
        n_steps = 0
        for i_episode in range(n_training_episodes):
            observation, _ = env.reset()
            if self.n_notes is not None:
                self.note_array = np.zeros(self.n_notes, dtype=np.float32)
            done = False
            t = 0
            episode_reward = 0

            for t in range(max_t):
                action, log_prob, value, obs_input = self.choose_action(observation)
                observation_, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                self.memory.store_memory(obs_input, log_prob, action, value, reward, done)
                n_steps += 1
                episode_reward += reward
                observation = observation_

                if n_steps % self.N == 0:
                    self._learn()

                if done:
                    break

            self.episode_rewards.append(episode_reward)
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

        if len(rewards_t) >= 100:
            means = rewards_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.0001)

    def eval(self):
        self.actor.eval()
        self.critic.eval()

    def train_mode(self):
        self.actor.train()
        self.critic.train()

    def select_action(self, state, deterministic=False):
        with torch.no_grad():
            if isinstance(state, torch.Tensor):
                obs = state.float().to(self.device)
            else:
                obs = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)
            if self.n_notes is not None:
                notes = torch.from_numpy(self.note_array).float().unsqueeze(0).to(self.device)
                obs = torch.cat((obs, notes), dim=1)
                actor_out = self.actor(obs)
                new_note = actor_out[1]
                gate_probs = actor_out[2] if self.write_gate is not None else None
                self._apply_note_update(new_note, gate_probs)
            else:
                actor_out = self.actor(obs)
            dist = actor_out if self.n_notes is None else actor_out[0]
            if deterministic:
                return dist.probs.argmax().item()
            return dist.sample().item()

    def save_model(self, filepath):
        torch.save({
            'agent': 'ppo',
            'config': self._config,
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'episode_rewards': self.episode_rewards,
        }, filepath)
        print(f'Model saved to: {filepath}')

    # Deprecated: Use from_checkpoint() instead for loading with config
    def load_model(self, filepath):
        checkpoint = torch.load(filepath, weights_only=False)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.episode_rewards = checkpoint['episode_rewards']
        print(f'Model loaded from: {filepath}')

    @classmethod
    def from_checkpoint(cls, filepath, device=None):
        checkpoint = torch.load(filepath, map_location=device, weights_only=False)
        if 'config' not in checkpoint:
            raise ValueError(f"Checkpoint '{filepath}' has no config. Use load_model() with a manually constructed agent instead.")
        agent = cls(**checkpoint['config'], device=device)
        agent.actor.load_state_dict(checkpoint['actor_state_dict'])
        agent.critic.load_state_dict(checkpoint['critic_state_dict'])
        agent.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        agent.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        agent.episode_rewards = checkpoint['episode_rewards']
        print(f'Model loaded from: {filepath}')
        return agent
