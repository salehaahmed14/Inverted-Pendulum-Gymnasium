from stable_baselines3 import PPO
import matplotlib.pyplot as plt
from custom_env import CustomInvertedPendulum
from custom_callback import CustomCallback

# define n_steps values for search
n_steps_values = [512, 1024, 2048]

#result variables for plotting
all_rewards = {n_steps: [] for n_steps in n_steps_values}
all_losses = {n_steps: [] for n_steps in n_steps_values}

# train and collect data for each n_steps
for n_steps in n_steps_values:
    # instantiate the custom environment
    env = CustomInvertedPendulum()
    
    # initialize PPO model with default parameters and varying n_steps
    model = PPO(
        policy="MlpPolicy",
        env=env,
        n_steps=n_steps,
        verbose=1
    )

    # initialize the callback with the current n_steps value
    callback = CustomCallback(n_steps)

    # train the model
    model.learn(total_timesteps=100000, callback=callback)

    # store results for plotting
    all_rewards[n_steps] = callback.rewards
    all_losses[n_steps] = callback.losses

# plot episodic rewards
plt.subplot(1, 2, 1)
for n_steps in n_steps_values:
    plt.plot(all_rewards[n_steps], label=f"n_steps {n_steps}")
plt.xlabel('Episodes')
plt.ylabel('Episodic Rewards')
plt.title('Training Rewards for Different n_steps Values')
plt.legend()

# plot total loss
plt.subplot(1, 2, 2)
for n_steps in n_steps_values:
    plt.plot(all_losses[n_steps], label=f"n_steps {n_steps}")
plt.xlabel('Number of Updates')
plt.ylabel('Total Loss Per Update')
plt.title('Total Loss for Different n_steps Values')
plt.legend()

plt.tight_layout()
plt.show()