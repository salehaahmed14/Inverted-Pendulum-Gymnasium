from stable_baselines3 import PPO
import matplotlib.pyplot as plt
from custom_env import CustomInvertedPendulum
from custom_callback import CustomCallback

# define the gamma values to test
gamma_values = [0.9, 0.95, 0.99]

#result variables for plotting
all_rewards = {gamma: [] for gamma in gamma_values}
all_losses = {gamma: [] for gamma in gamma_values}

# train and collect data for each gamma value
for gamma in gamma_values:
    # instantiate the custom environment
    env = CustomInvertedPendulum()
    
    # initialize PPO model with varying gamma
    model = PPO(
        policy="MlpPolicy",
        env=env,
        gamma=gamma,
        verbose=1
    )

    # initialize the callback
    callback = CustomCallback()

    # train the model
    model.learn(total_timesteps=100000, callback=callback)

    # store results for plotting
    all_rewards[gamma] = callback.rewards
    all_losses[gamma] = callback.losses

# plot episodic rewards
plt.subplot(1, 2, 1)
for gamma in gamma_values:
    plt.plot(all_rewards[gamma], label=f"Gamma {gamma}")
plt.xlabel('Episodes')
plt.ylabel('Episodic Rewards')
plt.title('Training Rewards for Different Gamma Values')
plt.legend()

# plot total loss
plt.subplot(1, 2, 2)
for gamma in gamma_values:
    plt.plot(all_losses[gamma], label=f"Gamma {gamma}")
plt.xlabel('Number of Updates')
plt.ylabel('Total Loss Per Update')
plt.title('Total Loss for Different Gamma Values')
plt.legend()

plt.tight_layout()
plt.show()
