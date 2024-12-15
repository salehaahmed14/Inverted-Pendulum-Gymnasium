from stable_baselines3 import PPO
import matplotlib.pyplot as plt
from custom_env import CustomInvertedPendulum
from custom_callback import CustomCallback

# Define n_steps values for search
n_steps_values = [512, 1024, 2048]

# Store results for plotting
all_rewards = {n_steps: [] for n_steps in n_steps_values}
all_losses = {n_steps: [] for n_steps in n_steps_values}

# Train and collect data for each n_steps
for n_steps in n_steps_values:
    # Instantiate the custom environment
    env = CustomInvertedPendulum()
    
    # Initialize PPO model with default parameters and varying n_steps
    model = PPO(
        policy="MlpPolicy",
        env=env,
        n_steps=n_steps,
        verbose=1
    )

    # Initialize the callback with the current n_steps value
    callback = CustomCallback(n_steps)

    # Train the model
    model.learn(total_timesteps=100000, callback=callback)

    # Store results for plotting
    all_rewards[n_steps] = callback.rewards
    all_losses[n_steps] = callback.losses

# Plot rewards
plt.subplot(1, 2, 1)
for n_steps in n_steps_values:
    plt.plot(all_rewards[n_steps], label=f"n_steps {n_steps}")
plt.xlabel('Episodes')
plt.ylabel('Episodic Rewards')
plt.title('Training Rewards for Different n_steps Values')
plt.legend()

# Plot total loss
plt.subplot(1, 2, 2)
for n_steps in n_steps_values:
    plt.plot(all_losses[n_steps], label=f"n_steps {n_steps}")
plt.xlabel('Number of Updates')
plt.ylabel('Total Loss Per Update')
plt.title('Total Loss for Different n_steps Values')
plt.legend()

# Adjust layout to make sure the titles are visible
plt.tight_layout()
plt.show()