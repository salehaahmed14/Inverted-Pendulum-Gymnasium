
# Inverted Pendulum Solver using PPO in MuJoCo

This repository provides an implementation of the **Proximal Policy Optimization (PPO)** algorithm to solve the **Inverted Pendulum** problem using the **MuJoCo** physics engine. The project demonstrates reinforcement learning applied to control problems, showcasing stable and efficient training with PPO.

## Features
- **Environment**: MuJoCo's `InvertedPendulum-v5` environment.
- **Algorithm**: PPO, a policy-gradient method for continuous control tasks.
- **Implementation**: Built with Stable-Baselines3 and OpenAI Gymnasium.
- **Visualization**: Training rewards and pendulum behavior during and after training.

## Requirements
To run the code, ensure you have the following installed:
- Python 3.8+
- PyTorch
- NumPy
- Matplotlib

## Usage

### 1. Clone the Repository
```bash
git clone https://github.com/salehaahmed14/Inverted-Pendulum-Gymnasium.git
```

### 2. Install Dependencies
Install dependencies using:
```bash
pip install swig
```
```bash
pip install gymnasium[all]
```
```bash
pip install stable-baselines3[extra]
```

### 2. Optimal Hyperparameters
The repository contains code files to determine the optimal values for gamma, learning rate, n_steps and clip range. That can be run using:
```bash
python PPO_optimal_gamma_search.py
```
```bash
python PPO_optimal_learning_rate_search.py
```
```bash
python PPO_optimal_n_steps_search.py
```
```bash
python PPO_optimal_clip_range_search.py
```

### 3. Train the Agent
Run the training script:
```bash
python PPO_main.py
```
This will save the trained model within the same directory.
- The agent successfully stabilizes the inverted pendulum.
- Training performance is logged and visualized, including:
  - Rewards over episodes.
  - Episodic Losses.

### 4. Evaluate the Agent
Test the trained model:
```bash
python PPO_evaluation.py
```
This will load the saved model and display the pendulum's performance by rendering the environment where the pendulum can seen being balanced.

## Results
The agent converges to the maximum reward of 1000 in 500 episodes using PPO.
