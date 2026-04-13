import gymnasium as gym
import numpy as np


class PartialObsCartPoleWrapper(gym.ObservationWrapper):
    """
    Wrapper for CartPole-v1 that hides Cart Velocity and Pole Angular Velocity.
    """
    
    def __init__(self, env):
        super().__init__(env)
        # Update the observation space to only include Cart Position and Pole Angle
        original_space = env.observation_space
        self.observation_space = gym.spaces.Box(
            low=np.array([original_space.low[0], original_space.low[2]]),
            high=np.array([original_space.high[0], original_space.high[2]]),
            dtype=np.float32
        )
    
    def observation(self, obs):
        """Return only Cart Position (index 0) and Pole Angle (index 2)."""
        return np.array([obs[0], obs[2]], dtype=np.float32)


def make_partial_obs_cartpole(render_mode=None):
    """Factory function to create the wrapped CartPole environment."""
    env = gym.make("CartPole-v1", render_mode=render_mode)
    return PartialObsCartPoleWrapper(env)
