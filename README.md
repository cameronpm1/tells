# Multi-Agent Training Environment (Communications-Denied Scenarios)

## Overview
This repository provides a training environment for collaborative multi-agent behavior in communications-denied environments.

## Current Support
- MARL (Multi-Agent Reinforcement Learning) codebase
- Uses RLlib with Soft Actor-Critic (SAC) or PPO for training
- Supports both MPE2 predator prey environment and pybullet-drones training environment


The default sync installs the predator-prey/MARL stack. The pybullet drone
environment is optional; install it only when needed:

```bash
git submodule update --init --recursive

pip install -e external/pybullet-drones
```

---

## Usage

### Training
```bash
python run.py --command marl_train --config confs/predator_prey/3a_game.yaml
```
or 
```bash
python run.py --command marl_train --config confs/drones/3a_game.yaml
```
- Starts training using the specified configuration file  
- Logs are stored in: `logs/marl/` (will autogenerate the directory if not already created)

---

### Evaluation
```bash
python run.py --command marl_eval --config confs/predator_prey/3a_game.yaml --model_dir logs/marl/test12/checkpoint3500 --runs 1
```
- Evaluates a trained checkpoint for a specified number of runs  
- Generates video outputs  
- Saves videos in a `video/` folder inside the checkpoint directory  

---

### Tensorboard 
```bash
tensorboard --logdir logs/marl/test12/
```
- Shows tensorboard log files for specified directory 

---

## Important Class Structures (predator_prey)

### Environment Framework
- The MARL framework uses the `PredatorPreyEnv` located in:
  ```
  envs/marl/
  ```

### Scenario Design
- Base scenario class: `PredatorPreyScenario`  
- Built on the MPE2 codebase  

#### Simulation Details
- 2D point-mass environment  
- N agents  
- 1 target  

### Target Behavior
- Controlled via a potential fields algorithm  
- Computes opposing forces to guide target movement  
- Objective, agents must corral the target toward a predefined landmark  

### Episode Cutoffs
- Max episode steps reached
- Target or agent location outside of GRID_SIZE (defined in env file)

### Environment Composition
- `PredatorPreyScenario` is wrapped inside:
  ```
  ScenarioEnv
  ```
  - Custom MPE2 container class  

#### Full Environment Stack
```
PredatorPreyEnv (Gym Environment)
    └── ScenarioEnv
            └── PredatorPreyScenario
```

### RLlib Integration
- `PredatorPreyEnv` is wrapped with:
  ```
  RLlibWrapper
  ```
- Ensures compatibility with RLlib MultiAgentEnv interface  
- Used for both training and evaluation pipelines  

## Important Class Structures (drones)

### Environment Framework

- The pybullet-drones environment is handled by DroneEnv, and uses similar rules to the predator_prey environment but adapted to a 3D drone framework
- The DroneEnv is wrapped inside RlLibWrapper so that the same training script can be used for both environments

## Training a Belief Model

### Data Collection

- Data for the belief model is stored inside a data/ directory
-Data collection can be run by using the collect_data command:

```bash
python run.py --command marl_collect_data --config confs/predator_prey/3a_game.yaml --model_dir logs/marl/ppotest1/checkpoint10000 --runs 800 --n_workers 8
```

- runs is the number of epsisodes to collect data from

- n_workers is the number of parallel environments to create when collecting data (important unless you want to wait a while)

- currently, collect data is only supported for predator_prey class, and for each timestep will log a timeseries of the preys behavior leading up to that timestep in relation to a randomly selected predator A, as well as the state of the other predators in relation to A

### Training a Model

- Once data is collected, a model can be trained by running the following command: 

```bash
python run.py --command belief_train --config confs/belief/test1.yaml
```

- Training data is loaded with the custom_dataset class inside learn/belief/custom_dataset.py. It is important to make sure that the .get_data() function properly loads in data

### Evaluating a Model

- Model evaluation can be done by running:

```bash
python run.py --command belief_eval --model_dir logs/belief/_ppotest1_NN_ppotest2/lightning_logs/checkpoints/{epoch}_{val_loss} --config confs/belief/test1.yaml
```

- This will create a directory called test_outputs inside the model_dir and save several test videos to it. In order to edit how the videos are created/saved look at .test_save() inside the belief model class

### Model Parameters

- Model input and output parameters are defined inside the config file, as well as training parameters (lr, epochs, etc.)

- To edit model architecture/loss function see models.py inside learn/belief

## Testing Policy with a Belief Model

```bash
python run.py --command marl_eval_belief --belief_dir logs/belief/_ppo_bc_noisierinject_NN_ppo_bc_test3/lightning_logs/checkpoints/{epoch}_{val_loss} --belief_config confs/belief/test1.yaml --config confs/predator_prey/3a_game.yaml --model_dir logs/marl/ppo_bc_ex/checkpoint4500/
```

## Training a Policy with a Belief Model

```bash
python run.py --command marl_train_belief --belief_dir logs/belief/_ppo_bc_noisierinject_NN_ppo_bc_test3/lightning_logs/checkpoints/{epoch}_{val_loss} --belief_config confs/belief/test1.yaml --config confs/predator_prey/3a_game.yaml
```

## Baselines

### Test Particle Filter

```bash
python run.py --command pf_eval --model_dir logs/marl/ppo_bc/checkpoint2000 --config confs/predator_prey/3a_game.yaml
```

---

## Notes
- Designed for experimentation in decentralized, communication-restricted coordination tasks  
- Modular structure allows extension to additional scenarios and agent behaviors  
