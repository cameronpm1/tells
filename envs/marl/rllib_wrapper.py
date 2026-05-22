from copy import deepcopy
from ray.rllib.env.multi_agent_env import MultiAgentEnv


class RLLibWrapper(MultiAgentEnv):
    '''
    wrapper for multiagent envs
    env should handle dict format, wrapper is
    for rllib MultiAgentEnv inheritance
    '''

    def __init__(
            self,
            env,
            eval: bool = False,
            belief: bool = False,
    ): 
        super().__init__()

        self.env = env
        self.eval = eval
        self.belief = belief
        self.agents = deepcopy(env.agents)
        self.possible_agents = deepcopy(env.agents)
        self.last_raw_reward = None

        self.single_observation_spaces = {
            agent: self.env._observation_space(agent)
            for agent in self.agents
        }

        self.single_action_spaces = {
            agent: self.env._action_space(agent)
            for agent in self.agents
        }

        # disable deprecated paths
        self.observation_spaces = None
        self.action_spaces = None

        self.observation_space = None
        self.action_space = None

    def get_observation_space(self, agent):
        return self.single_observation_spaces[agent]

    def get_action_space(self, agent):
        return self.single_action_spaces[agent]
    
    def step(self, action_dict):
        obs,rew,terminated,truncated,raw_infos = self.env.step(action_dict)
        rew = dict(rew)
        raw_infos = raw_infos if isinstance(raw_infos, dict) else {}
        infos = {
            agent: dict(raw_infos.get(agent, {}))
            for agent in self.agents
        }

        obs.pop("target", None)
        rew.pop('target', None)
        terminated.pop('target', None)
        truncated.pop('target', None)

        terminated["__all__"] = all(terminated.values())
        truncated["__all__"] = all(truncated.values())
        #print(obs.keys(),rew.keys(),terminated.keys(),truncated.keys(),_.keys())
        
        if self.belief:
            for agent in self.agents:
                obs[agent+'_partial'] = obs[agent][4:8]
        
        infos['__common__'] = {}
        infos['__common__']['raw_reward'] = sum(rew.values())
        self.last_raw_reward = sum(rew.values())

        return obs,rew,terminated,truncated,infos
    
    def reset(self, **kwargs):
        obs,infos = {},{}
        obs,infos = self.env.reset(**kwargs)

        infos['__common__'] = {}
        infos['__common__']['raw_reward'] = 0.0
        self.last_raw_reward = 0.0

        if self.belief:
            for agent in self.agents:
                obs[agent+'_partial'] = obs[agent][4:8]

        return obs,infos
    
    def close(self):
        close_fn = getattr(self.env, 'close', None)
        if close_fn is not None:
            close_fn()

    def render_rgb(self):
        return self.env.render_rgb()

    def set_difficulty(self, difficulty):
        self.env.set_difficulty(difficulty)
