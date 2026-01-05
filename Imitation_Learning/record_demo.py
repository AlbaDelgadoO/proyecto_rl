import pickle
from trex_env_imitation import DinoEnv, HumanPolicy

expert_data_path = "expert_demos.pkl"

def main():
    env = DinoEnv(render_mode="human")
    policy = HumanPolicy()

    # Trackers for the best performance
    best_score = -1
    best_trajectory = []
    
    # Current episode buffer
    current_episode_data = []

    print("Controls: ↑ / SPACE = jump, ↓ = duck (hold)")
    print("Play as many times as you want.")
    print("Press Ctrl+C in the terminal to stop and save the BEST run.")

    obs, _ = env.reset()

    try:
        while True:
            # Get action from human
            action = policy.get_action(obs)
            
            # Handle window close (X button)
            if action is None:
                break

            next_obs, reward, done, truncated, info = env.step(action)
            env.render()
            
            # Store the step (observation, action)
            current_episode_data.append((obs, action))
            obs = next_obs

            # If episode ends
            if done or truncated:
                final_score = env.points
                print(f"Episode finished. Score: {final_score}")

                # Check if this is the new best record
                if final_score > best_score:
                    best_score = final_score
                    # Save a copy of the list so it doesn't get overwritten
                    best_trajectory = list(current_episode_data)
                    print(f">>> New High Score! (Temporarily saved in memory)")
                else:
                    print(f"--- Did not beat high score of {best_score}")

                # Reset for the next game
                current_episode_data = []
                obs, _ = env.reset()

    except KeyboardInterrupt:
        print("\nInterrupted by user (Ctrl+C). Stopping...")

    finally:
        # Save logic runs whether you press Ctrl+C or close the window
        if best_trajectory:
            print(f"Saving the BEST episode (Score: {best_score}) to {expert_data_path}...")
            # We wrap it in a list [best_trajectory] to maintain the expected structure 
            # (a list of episodes) for the training script.
            with open(expert_data_path, "wb") as f:
                pickle.dump([best_trajectory], f)
            print("Saved successfully.")
        else:
            print("No complete episodes recorded. Nothing saved.")
        
        env.close()

if __name__ == "__main__":
    main()