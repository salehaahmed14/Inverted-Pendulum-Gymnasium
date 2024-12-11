import gymnasium as gym
from stable_baselines3 import DDPG
import numpy as np
import optuna


env = gym.make("InvertedPendulum-v5")

def objective(trial):
    # Sample hyperparameters from a distribution
    gamma = trial.suggest_float('gamma', 0.9, 0.99)
    tau = trial.suggest_float('tau', 0.001, 0.01)
    batch_size = trial.suggest_int('batch_size', 32, 512, step=32)
    buffer_size = trial.suggest_int('buffer_size', 10000, 1000000, step=10000)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)

    # Initialize DDPG model with suggested hyperparameters
    model = DDPG(
        'MlpPolicy', env,
        gamma=gamma,
        tau=tau,
        batch_size=batch_size,
        buffer_size=buffer_size,
        learning_rate=learning_rate,
        verbose=1  # Verbose output for early training
    )

    # Training the model
    print("Training model...")
    model.learn(total_timesteps=10000, reset_num_timesteps=False)

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