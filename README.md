# Multi-Agent Training Environment (Communications-Denied Scenarios)

## Overview
This repository provides a training environment for collaborative multi-agent behavior in communications-denied environments.

## Current Support
- MARL (Multi-Agent Reinforcement Learning) codebase
- Uses RLlib with Soft Actor-Critic (SAC) for training

---

## Usage

### Training
```bash
python run.py --command marl_train --config confs/predator_prey/3a_game.yaml
```
- Starts training using the specified configuration file  
- Logs are stored in: `logs/marl/` (will autogenerate the directory if not already created)

---

### Evaluation
```bash
python run.py --command marl_eval --config confs/predator_prey/3a_game.yaml --model_dir /home/cameron/tells/logs/marl/test12/checkpoint3500 --runs 1
```
- Evaluates a trained checkpoint for a specified number of runs  
- Generates video outputs  
- Saves videos in a `video/` folder inside the checkpoint directory  

---

### Tensorboard 
```bash
python tensorboard --logdir logs/marl/test12/
```
- Shows tensorboard log files for specified directory 

---

## Important Class Structures

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

---

## Notes
- Designed for experimentation in decentralized, communication-restricted coordination tasks  
- Modular structure allows extension to additional scenarios and agent behaviors  
