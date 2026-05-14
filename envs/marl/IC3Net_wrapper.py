import torch
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class IC3NetWrapper(gym.Env):
    """
    Old-Gym adapter for the original IC3Net codebase.

    Input:
        predator_prey_env:
            An initialized PredatorPreyEnv instance.

    Output API expected by IC3Net:
        reset() -> tuple/list of per-agent observations
        step(actions) -> obs, reward, done, info

    This wrapper assumes:
        - The target/adversary is controlled internally
        - Only predator agents are trained.
        - actions contains one discrete action per predator.
    """

    def __init__(self, predator_prey_env):
        super().__init__()

        self.env = predator_prey_env
        self.agent_names = list(self.env.agents)
        self.nagents = len(self.agent_names)

        self.t = 0
        self.stat = {}

        self.max_steps = self.env.unwrapped.max_cycles if hasattr(self.env.unwrapped, "max_cycles") else None

        sample_obs_dict, _ = self.env.reset()
        sample_obs = self._pack_obs(sample_obs_dict)

        if len(sample_obs) == 0:
            raise RuntimeError("IC3NetWrapper received zero trainable agents.")

        obs_dim = int(np.asarray(sample_obs[0], dtype=np.float32).reshape(-1).shape[0])

        for i, obs in enumerate(sample_obs):
            this_dim = int(np.asarray(obs, dtype=np.float32).reshape(-1).shape[0])
            if this_dim != obs_dim:
                raise RuntimeError(
                    "IC3Net requires all agents to have the same flattened observation dimension. "
                    f"Agent 0 has dim {obs_dim}, but agent {i} has dim {this_dim}."
                )

        sample_obs_dict, _ = self.env.reset()
        sample_obs = self._pack_obs(sample_obs_dict)

        if sample_obs.ndim != 3:
            raise RuntimeError(
                f"Expected packed observation shape (1, nagents, obs_dim), got {tuple(sample_obs.shape)}"
            )

        if sample_obs.shape[1] != self.nagents:
            raise RuntimeError(
                f"Expected {self.nagents} agent observations, got {sample_obs.shape[1]}"
            )

        obs_dim = int(sample_obs.shape[2])

        self.observation_dim = obs_dim
        self.num_inputs = obs_dim
        self.dim = obs_dim

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=tuple(sample_obs.shape),
            dtype=np.float64,
        )

        first_agent = self.agent_names[0]
        self.action_space = self.env._action_space(first_agent)
        if isinstance(self.action_space, spaces.Discrete):
            self.num_actions = int(self.action_space.n)
            self.nactions = int(self.action_space.n)
            self.dim_actions = 1
        elif isinstance(self.action_space, spaces.Box):
            self.num_actions = int(np.prod(self.action_space.shape))
            self.nactions = self.num_actions
            self.dim_actions = self.num_actions


    def reset(self, epoch=None):
        self.t = 0

        obs_dict, _ = self.env.reset()
        obs = self._pack_obs(obs_dict)

        return obs

    def step(self, actions):
        self.t += 1

        action_dict = self._unpack_actions(actions)

        obs_dict, rewards_dict, terminations, truncations, infos = self.env.step(action_dict)

        obs = self._pack_obs(obs_dict)

        rewards = np.array(
            [float(rewards_dict.get(agent, 0.0)) for agent in self.agent_names],
            dtype=np.float64,
        )

        done = bool(
            terminations.get("__all__", False)
            or truncations.get("__all__", False)
            or (self.max_steps is not None and self.t >= self.max_steps)
        )

        info = {
            "per_agent_info": {
                agent: infos.get(agent, {}) for agent in self.agent_names
            }
        }

        if self.agent_names:
            first_agent_info = infos.get(self.agent_names[0], {})
            if isinstance(first_agent_info, dict):
                info.update(first_agent_info)

        return obs, rewards, done, info

    def _pack_obs(self, obs_dict):
        """
        IC3Net expects a torch tensor of shape:

            (nagents, observation_dim)

        because trainer.py passes state directly into policy_net.
        """

        packed = []

        for agent in self.agent_names:
            if agent not in obs_dict:
                raise KeyError(
                    f"Missing observation for trainable agent '{agent}'. "
                    f"Available observation keys: {list(obs_dict.keys())}"
                )

            obs = np.asarray(obs_dict[agent], dtype=np.float64).reshape(-1)
            packed.append(obs)

        if len(packed) == 0:
            raise RuntimeError("No observations were packed for IC3Net.")

        packed = np.stack(packed, axis=0)      # (nagents, obs_dim)
        packed = packed[None, :, :]            # (1, nagents, obs_dim)

        return torch.as_tensor(packed, dtype=torch.get_default_dtype())

    def _unpack_actions(self, actions):
        """
        Converts IC3Net translate_action(...) output into the dictionary expected by
        the wrapped PredatorPreyEnv.

        IC3Net may pass actions as:
            - array shape (nagents,)
            - array shape (1, nagents)
            - array shape (nagents, 1)
            - list/tuple where actions[0] is the physical action per agent
            - multi-action output where only the first action dimension is the env action

        For this wrapper, only the first physical action dimension is sent to the env.
        """

        import torch

        if torch.is_tensor(actions):
            actions = actions.detach().cpu().numpy()

        # Common IC3Net case: actual is a list of action dimensions.
        # Example:
        #   actions[0] = physical action for each agent
        #   actions[1] = communication / auxiliary action
        if isinstance(actions, (list, tuple)):
            if len(actions) == 0:
                raise ValueError("Received empty action list from IC3Net.")

            first = actions[0]
            if torch.is_tensor(first):
                first = first.detach().cpu().numpy()

            first = np.asarray(first).reshape(-1)

            if first.size == self.nagents:
                physical_actions = first
            else:
                arr = np.asarray(actions)
                physical_actions = self._extract_physical_actions_from_array(arr)

        else:
            arr = np.asarray(actions)
            physical_actions = self._extract_physical_actions_from_array(arr)

        if physical_actions.size != self.nagents:
            raise ValueError(
                f"Expected {self.nagents} physical actions, got {physical_actions.size}. "
                f"Raw IC3Net actions: {actions}"
            )

        action_dict = {}

        for i, agent in enumerate(self.agent_names):
            action = int(physical_actions[i])

            if isinstance(self.action_space, spaces.Discrete):
                if action < 0 or action >= int(self.action_space.n):
                    raise ValueError(
                        f"Invalid action {action} for agent {agent}. "
                        f"Expected Discrete({self.action_space.n}). "
                        f"Raw IC3Net actions: {actions}"
                    )

            action_dict[agent] = action

        return action_dict


    def _extract_physical_actions_from_array(self, arr):
        """
        Extracts one physical action per agent from an IC3Net action array.
        """

        arr = np.asarray(arr)

        if arr.ndim == 0:
            arr = arr.reshape(1)

        # Already one action per agent.
        if arr.ndim == 1 and arr.size == self.nagents:
            return arr.reshape(self.nagents)

        # Shape: (dim_actions, nagents), use first action dimension.
        if arr.ndim == 2 and arr.shape[1] == self.nagents:
            return arr[0, :].reshape(self.nagents)

        # Shape: (nagents, dim_actions), use first column.
        if arr.ndim == 2 and arr.shape[0] == self.nagents:
            return arr[:, 0].reshape(self.nagents)

        # Last-resort handling for flattened outputs.
        flat = arr.reshape(-1)

        if flat.size == self.nagents:
            return flat

        if flat.size % self.nagents == 0:
            reshaped = flat.reshape(-1, self.nagents)

            # Treat rows as action dimensions and columns as agents.
            # The first row is the physical action dimension.
            return reshaped[0, :].reshape(self.nagents)

        raise ValueError(
            f"Could not extract physical actions for {self.nagents} agents. "
            f"Raw shape: {arr.shape}, raw values: {arr}"
        )

    def render_rgb(self):
        return self.env.render_rgb()

    def close(self):
        if hasattr(self.env, "close"):
            self.env.close()