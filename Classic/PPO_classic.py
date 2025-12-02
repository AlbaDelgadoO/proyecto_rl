import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from trex_env import DinoEnv  # tu environment

# -----------------------------
# Directorios para modelos y logs
# -----------------------------
ALGORITHM = "PPO"
models_dir = f"models/{ALGORITHM}"
log_dir = "logs"

os.makedirs(models_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# -----------------------------
# Crear el environment con Monitor
# -----------------------------
env = DinoEnv()
env = Monitor(env, log_dir)

# -----------------------------
# Buscar último modelo guardado
# -----------------------------
checkpoints = [f for f in os.listdir(models_dir) if f.endswith(".zip")]
if checkpoints:
    latest_model_path = os.path.join(models_dir, sorted(checkpoints, key=lambda x: int(x.split(".zip")[0]))[-1])
    print(f"Cargando modelo existente: {latest_model_path}")
    model = PPO.load(latest_model_path, env=env)
else:
    print("No se encontró modelo previo, creando uno nuevo")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99
    )

# -----------------------------
# Entrenamiento
# -----------------------------
TIMESTEPS = 50000 
NUM_ITERATIONS = 10

for i in range(1, NUM_ITERATIONS + 1):
    model.learn(
        total_timesteps=TIMESTEPS,
        reset_num_timesteps=False,
        tb_log_name=ALGORITHM
    )
    model.save(f"{models_dir}/{TIMESTEPS * i}")
    print(f"Modelo guardado: {models_dir}/{TIMESTEPS * i}")

# -----------------------------
# Evaluación del agente
# -----------------------------
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")

env.close()