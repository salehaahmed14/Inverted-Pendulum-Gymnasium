from stable_baselines3 import PPO
import matplotlib.pyplot as plt
from custom_env import CustomInvertedPendulum
from custom_callback import CustomCallback

# Define the gamma values to test
gamma_values = [0.9, 0.95, 0.99]

# Store results for plotting
all_rewards = {gamma: [] for gamma in gamma_values}
all_losses = {gamma: [] for gamma in gamma_values}

# Train and collect data for each gamma value
for gamma in gamma_values:
    # Instantiate the custom environment
    env = CustomInvertedPendulum()
    
    # Initialize PPO model with varying gamma
    model = PPO(
        policy="MlpPolicy",
        env=env,
        gamma=gamma,  # Varying the gamma value
        verbose=1
    )

    # Initialize the callback
    callback = CustomCallback()

    # Train the model
    model.learn(total_timesteps=100000, callback=callback)

    # Store results for plotting
    all_rewards[gamma] = callback.rewards
    all_losses[gamma] = callback.losses

# Plot rewards
plt.subplot(1, 2, 1)
for gamma in gamma_values:
    plt.plot(all_rewards[gamma], label=f"Gamma {gamma}")
plt.xlabel('Episodes')
plt.ylabel('Episodic Rewards')
plt.title('Training Rewards for Different Gamma Values')
plt.legend()

# Plot total loss
plt.subplot(1, 2, 2)
for gamma in gamma_values:
    plt.plot(all_losses[gamma], label=f"Gamma {gamma}")
plt.xlabel('Number of Updates')
plt.ylabel('Total Loss Per Update')
plt.title('Total Loss for Different Gamma Values')
plt.legend()

# Adjust layout to make sure the titles are visible
plt.tight_layout()
plt.show()
