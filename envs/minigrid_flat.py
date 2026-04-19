import gymnasium as gym
import numpy as np

# Remap raw object indices to a small vocabulary for MemoryS7:
#   unseen(0), empty(1) -> 0  (nothing)
#   wall(2)             -> 1
#   key(5)              -> 2
#   ball(6)             -> 3
#   agent(10)           -> 4
#   everything else     -> 0
_OBJ_REMAP = {0: 0, 1: 0, 2: 1, 5: 2, 6: 3, 10: 4}
_OBJ_REMAP_MAX = 4.0  # normalise into [0, 1]


class MiniGridFlatWrapper(gym.ObservationWrapper):
    """Flattens MiniGrid dict obs into a compact 1D float array.

    Only the object-type channel of the 7x7 image is kept (colour and
    door-state are irrelevant for MemoryS7).  Object indices are remapped
    to a small vocabulary and normalised.  Direction is appended as a
    single normalised value.  Total: 7*7 + 1 = 50 features.
    """

    def __init__(self, env):
        super().__init__(env)
        n = 7 * 7 + 1  # 50
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(n,), dtype=np.float32)

    def observation(self, obs):
        obj_channel = obs['image'][:, :, 0]  # (7, 7) raw object indices
        remapped = np.vectorize(_OBJ_REMAP.get)(obj_channel, 0).astype(np.float32)
        image = remapped.flatten() / _OBJ_REMAP_MAX
        direction = np.array([obs['direction'] / 3.0], dtype=np.float32)
        return np.concatenate([image, direction])


def make_minigrid_flat(env_name, render_mode=None):
    """Factory function to create a flat-obs MiniGrid environment."""
    import minigrid  # noqa: F401
    env = gym.make(env_name, render_mode=render_mode)
    return MiniGridFlatWrapper(env)
