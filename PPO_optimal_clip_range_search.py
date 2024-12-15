import gymnasium as gym
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
from custom_env import CustomInvertedPendulum
from custom_callback import CustomCallback

# Define clip_range values
clip_range_values = [0.1, 0.2, 0.3]

# Store results for plotting
all_rewards = {clip: [] for clip in clip_range_values}
all_losses = {clip: [] for clip in clip_range_values}

# Train and collect data for each clip_range
for clip_range in clip_range_values:
    # Instantiate the custom environment
    env = gym.make("InvertedPendulum-v5")
    custom_env = CustomInvertedPendulum(env)
    
    # Initialize PPO model with default parameters and varying clip_range
    model = PPO(
        policy="MlpPolicy",
        env=custom_env,
        clip_range=clip_range,
        verbose=1
    )

    # Initialize the callback
    callback = CustomCallback()

    # Train the model
    model.learn(total_timesteps=100000, callback=callback)

    # Store results for plotting
    all_rewards[clip_range] = callback.rewards
    all_losses[clip_range] = callback.losses

# Plot rewards
plt.subplot(1, 2, 1)
for clip_range in clip_range_values:
    plt.plot(all_rewards[clip_range], label=f"Clip Range {clip_range}")
plt.xlabel('Episodes')
plt.ylabel('Episodic Rewards')
plt.title('Training Rewards for Different Clip Range Values')
plt.legend()

# Plot total loss
plt.subplot(1, 2, 2)
for clip_range in clip_range_values:
    plt.plot(all_losses[clip_range], label=f"Clip Range {clip_range}")
plt.xlabel('Number of Updates')
plt.ylabel('Total Loss Per Update')
plt.title('Total Loss for Different Clip Range Values')
plt.legend()

# Adjust layout to make sure the titles are visible
plt.tight_layout()
plt.show()