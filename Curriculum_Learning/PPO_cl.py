import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from trex_env_cl2 import DinoEnv

# ======================================================
# Create environment with Monitor and curriculum phase
# ======================================================
def make_env(phase=1):
    env = DinoEnv(curriculum_phase=phase)
    env = Monitor(env)
    return env

# ======================================================
# Configuration
# ======================================================
checkpoint_dir = "./PPO_2/"
tensorboard_log = "./PPO_2_tensorboard/"

os.makedirs(checkpoint_dir, exist_ok=True)

# ======================================================
# Phase 1: Less obstacles (150k steps)
# ======================================================
print("=" * 50)
print("PHASE 1: Less obstacles")
print("=" * 50)

env = make_env(phase=1)

model = PPO(
    policy="MlpPolicy",
    env=env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    tensorboard_log=tensorboard_log,
    policy_kwargs=dict(net_arch=[512, 512, 256])
)

checkpoint_callback = CheckpointCallback(
    save_freq=20_000,
    save_path=checkpoint_dir,
    name_prefix="phase1"
)

model.learn(total_timesteps=150_000, callback=checkpoint_callback, progress_bar=True)
model.save(f"{checkpoint_dir}/phase1_final")
env.close()

# ======================================================
# Phase 2: Original environment (150k steps)
# ======================================================
print("=" * 50)
print("PHASE 2: Original environment")
print("=" * 50)

env = make_env(phase=2)
model.set_env(env)

checkpoint_callback = CheckpointCallback(
    save_freq=20_000,
    save_path=checkpoint_dir,
    name_prefix="phase2"
)

model.learn(total_timesteps=150_000, callback=checkpoint_callback, progress_bar=True, reset_num_timesteps=False)
model.save(f"{checkpoint_dir}/phase2_final")
env.close()

# ======================================================
# Phase 3: Birds at different heights (150k steps)
# ======================================================
print("=" * 50)
print("PHASE 3: Birds at different heights")
print("=" * 50)

env = make_env(phase=3)
model.set_env(env)

checkpoint_callback = CheckpointCallback(
    save_freq=20_000,
    save_path=checkpoint_dir,
    name_prefix="phase3"
)

model.learn(total_timesteps=150_000, callback=checkpoint_callback, progress_bar=True, reset_num_timesteps=False)
model.save(f"{checkpoint_dir}/phase3_final")
env.close()

print("=" * 50)
print("Curriculum Learning completed!")
print("=" * 50)