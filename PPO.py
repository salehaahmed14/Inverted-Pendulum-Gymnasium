import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np
import optuna


env = gym.make("InvertedPendulum-v5")

def objective(trial):
    # Sample hyperparameters for PPO
    gamma = trial.suggest_float('gamma', 0.9, 0.99)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    clip_range = trial.suggest_float('clip_range', 0.1, 0.3)
    n_steps_values = [64, 128, 256, 512, 1024, 2048]
    n_steps = trial.suggest_categorical('n_steps', n_steps_values)
    # batch_size = trial.suggest_int('batch_size', 32, 256, step=32)
    batch_size = n_steps

    model = PPO(
        'MlpPolicy', env,
        gamma=gamma,
        learning_rate=learning_rate,
        clip_range=clip_range,
        n_steps=n_steps,
        batch_size=batch_size,
        verbose=1
    )

    # Training the model
    print("Training model...")
    model.learn(total_timesteps=10000, reset_num_timesteps=False, progress_bar=True, log_interval=1000)

    # Evaluate the model to find the average reward
    rewards = []
    for _ in range(10):  # Run 10 evaluation episodes
        obs,_ = env.reset()
        terminated, truncated = False, False
        ep_reward = 0
        while not terminated and not truncated:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
        rewards.append(ep_reward)

    # Calculate the mean reward
    mean_reward = np.mean(rewards)
    print(f"Episode mean reward: {mean_reward}")

    # Return the mean reward as the objective to maximize
    return mean_reward

# Create an Optuna study object
study = optuna.create_study(direction='maximize')

# Perform the optimization with progress tracking using tqdm
print("Optimizing hyperparameters...")
study.optimize(objective, n_trials=50, n_jobs=1)

# Display the best hyperparameters found
print("Best hyperparameters found:", study.best_params)
print("Best average reward:", study.best_value)