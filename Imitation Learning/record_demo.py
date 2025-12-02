import pickle
from trex_env_imitation import DinoEnv, HumanPolicy

expert_data_path = "expert_demos.pkl"

def main():
    env = DinoEnv(render_mode="human")
    policy = HumanPolicy()

    trajectories = []

    print("Controls: ↑ / SPACE = jump, ↓ = duck (hold)")
    print("Press the X on the window to save and exit.")

    obs, _ = env.reset()
    episode = []

    while True:
        action = policy.get_action(obs)
        if action is None:  # Exit
            break

        next_obs, reward, done, truncated, info = env.step(action)
        env.render()
        episode.append((obs, action))
        obs = next_obs

        if done or truncated:
            print(f"Episode score: {env.points}")
            trajectories.append(episode)
            episode = []
            obs, _ = env.reset()

    with open(expert_data_path, "wb") as f:
        pickle.dump(trajectories, f)

    print(f"Data saved in {expert_data_path}")
    env.close()

if __name__ == "__main__":
    main()
