from stable_baselines3 import PPO
import matplotlib.pyplot as plt
from custom_env import CustomInvertedPendulum
from custom_callback import CustomCallback

# define the learning rates to test
learning_rates = [1e-5, 1e-4, 1e-3]

#result variables for plotting
all_rewards = {lr: [] for lr in learning_rates}
all_losses = {lr: [] for lr in learning_rates}

# train and collect data for each learning rate
for lr in learning_rates:
    # instantiate the custom environment
    env = CustomInvertedPendulum()
    
    # initialize PPO model with varying learning rate
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=lr,
        verbose=1
    )

    # initialize the callback
    callback = CustomCallback()

    # train the model
    model.learn(total_timesteps=100000, callback=callback)

    # store results for plotting
    all_rewards[lr] = callback.rewards
    all_losses[lr] = callback.losses

# plot episodic rewards
plt.subplot(1, 2, 1)
for lr in learning_rates:
    plt.plot(all_rewards[lr], label=f"Learning Rate {lr}")
plt.xlabel('Episodes')
plt.ylabel('Episodic Rewards')
plt.title('Training Rewards for Different Learning Rates')
plt.legend()

# plot total loss
plt.subplot(1, 2, 2)
for lr in learning_rates:
    plt.plot(all_losses[lr], label=f"Learning Rate {lr}")
plt.xlabel('Number of Updates')
plt.ylabel('Total Loss Per Update')
plt.title('Total Loss for Different Learning Rates')
plt.legend()

plt.tight_layout()
plt.show()
