import gymnasium as gym
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
from custom_env import CustomInvertedPendulum
from custom_callback import CustomCallback

# Define the learning rates to test
learning_rates = [1e-5, 1e-4, 1e-3]

# Store results for plotting
all_rewards = {lr: [] for lr in learning_rates}
all_losses = {lr: [] for lr in learning_rates}

# Train and collect data for each learning rate
for lr in learning_rates:
    # Instantiate the custom environment
    env = CustomInvertedPendulum()
    
    # Initialize PPO model with varying learning rate
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=lr,  # Varying the learning rate
        verbose=1
    )

    # Initialize the callback
    callback = CustomCallback()

    # Train the model
    model.learn(total_timesteps=100000, callback=callback)

    # Store results for plotting
    all_rewards[lr] = callback.rewards
    all_losses[lr] = callback.losses

# Plot rewards
plt.subplot(1, 2, 1)
for lr in learning_rates:
    plt.plot(all_rewards[lr], label=f"Learning Rate {lr}")
plt.xlabel('Episodes')
plt.ylabel('Episodic Rewards')
plt.title('Training Rewards for Different Learning Rates')
plt.legend()

# Plot total loss
plt.subplot(1, 2, 2)
for lr in learning_rates:
    plt.plot(all_losses[lr], label=f"Learning Rate {lr}")
plt.xlabel('Number of Updates')
plt.ylabel('Total Loss Per Update')
plt.title('Total Loss for Different Learning Rates')
plt.legend()

# Adjust layout to make sure the titles are visible
plt.tight_layout()
plt.show()
