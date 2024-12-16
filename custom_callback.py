from stable_baselines3.common.callbacks import BaseCallback

class CustomCallback(BaseCallback):
    def __init__(self, n_steps=2048):
        super(CustomCallback, self).__init__()
        self.rewards = []
        self.losses = []
        self.episode_reward = 0
        self.step_count = 0
        self.n_steps = n_steps

    def _on_step(self) -> bool:
        reward = self.locals.get('rewards')  #get the reward at the current step
        done = self.locals.get('dones')  #get the done status (whether the episode ended)
        
        if reward is not None:
            self.episode_reward += reward
        
        self.step_count += 1
        
        #log the loss after n_steps steps
        if self.step_count % self.n_steps == 0:
            total_loss = self.model.logger.name_to_value['train/loss']
            
            if total_loss is not None:
                self.losses.append(total_loss)

        if done:  #when an episode is done, store the reward and reset
            self.rewards.append(self.episode_reward)
            self.episode_reward = 0
            
        return True