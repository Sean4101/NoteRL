# NoteRL

A Deep RL agent that solves POMDPs by writing to an external "note" array at each timestep. The note is concatenated to the observation, giving the agent persistent working memory without recurrent layers.

Agents produce 2 separate outputs, $A^{(e)}$ for environment action and $A^{(n)}_t$ for "writing" a note. The observation is a concatenation of the environment observation and the note content: $O_t = (O^{(e)}_t, O^{(n)}_t)$.

![Architecture](assets/architecture.png)

## Project Structure

```
NoteRL/
├── agents/
│   ├── ppo.py              # PPO agent (classic, note, gated-note variants)
│   └── reinforce.py        # REINFORCE agent (classic, note variants)
├── envs/
│   └── partial_obs_cartpole.py   # CartPole with velocity observations hidden
├── configs/
│   ├── ppo_classic.yaml
│   ├── ppo_note.yaml
│   ├── ppo_gated.yaml
│   ├── reinforce_classic.yaml
│   └── reinforce_note.yaml
├── scripts/
│   ├── train.py            # Training script
│   └── play.py             # Evaluation / playback script
└── models/                 # Saved checkpoints
```

## Setup

**Requirements:** Python 3.10+

```bash
python -m venv .venv
.\.venv\Scripts\activate        # Windows
# source .venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
```

## Training

```bash
python scripts/train.py --config configs/ppo_note.yaml --save models/ppo_note.pth --n_episodes 10000
```

| Argument | Required | Description |
|---|---|---|
| `--config` | Yes | Path to a YAML config file |
| `--save` | Yes | Path to save the trained model |
| `--n_episodes` | No | Number of training episodes (default: 1000) |
| `--no_plot` | No | Disable the live training plot |

### Available configs

| Config | Agent | Notes | Write Gate |
|---|---|---|---|
| `ppo_classic.yaml` | PPO | — | — |
| `ppo_note.yaml` | PPO | 4 | No |
| `ppo_gated.yaml` | PPO | 4 | Yes |
| `reinforce_classic.yaml` | REINFORCE | — | — |
| `reinforce_note.yaml` | REINFORCE | 4 | — |

### Custom config

Copy any existing config and edit `agent_params`. The `env` field accepts:
- `CartPole-v1-partial` — cart position + pole angle only (POMDP)
- `CartPole-v1` — full observations

## Playing

```bash
python scripts/play.py --model models/ppo_note.pth --env CartPole-v1-partial
```

| Argument | Required | Description |
|---|---|---|
| `--model` | Yes | Path to a trained checkpoint (`.pth`) |
| `--env` | Yes | Environment name: `CartPole-v1` or `CartPole-v1-partial` |
| `--n_episodes` | No | Number of episodes to run (default: 10) |
| `--no_render` | No | Disable the renderer (faster evaluation) |

## Loading a trained model

Checkpoints saved by `train.py` include the full config, so no need to re-specify hyperparameters:

```python
from agents.ppo import PPOAgent

agent = PPOAgent.from_checkpoint('models/ppo_note.pth', device='cpu')
```

