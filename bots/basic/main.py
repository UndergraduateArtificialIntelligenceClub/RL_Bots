import rlgym
from terminal import CustomTerminalCondition
from reward import SpeedReward
from obs import CustomObsBuilderBluePerspective
from stable_baselines3 import PPO
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition

default_tick_skip = 8
physics_ticks_per_second = 120
ep_len_seconds = 20

max_steps = int(round(ep_len_seconds * physics_ticks_per_second / default_tick_skip))

# Make the default rlgym environment
env = rlgym.make(
    terminal_conditions=[CustomTerminalCondition(), TimeoutCondition(max_steps)],
    reward_fn=SpeedReward(),
    obs_builder=CustomObsBuilderBluePerspective(),
)

# Initialize PPO from stable_baselines3
model = PPO("MlpPolicy", env=env, verbose=1, device="cuda", learning_rate=0.001)

# Train our agent!
model.learn(total_timesteps=int(5 * 1e7))
model.save("rl_speed")
