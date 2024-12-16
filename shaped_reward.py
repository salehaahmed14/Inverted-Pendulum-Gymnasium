from stable_baselines3 import PPO
import matplotlib.pyplot as plt
from custom_env import CustomInvertedPendulum
from custom_callback import CustomCallback

#instantiate the custom environment
env = CustomInvertedPendulum()

# initialize PPO model with custom environment
model = PPO(
    policy="MlpPolicy",
    env=env,
    verbose=1
)

# initialize the callback
callback = CustomCallback()

#train the model
model.learn(total_timesteps=100000, callback=callback)

# plot episodic rewards
plt.subplot(1, 2, 1)
plt.plot(callback.rewards)
plt.xlabel('Episodes')
plt.ylabel('Episodic Rewards')
plt.title('Training Rewards')

# plot total loss
plt.subplot(1, 2, 2)
plt.plot(callback.losses)
plt.xlabel('Number of Updates')
plt.ylabel('Total Loss Per Update')
plt.title('Training Loss')

plt.tight_layout()
plt.show()