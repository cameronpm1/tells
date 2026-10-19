import numpy as np

N_LEVELS = 5


def level_to_production(action: int, capacity: float, n_levels: int = N_LEVELS) -> float:
    return capacity * int(action) / (n_levels - 1)

def production_to_level(production: float, capacity: float, n_levels: int = N_LEVELS) -> int:
    return int(np.clip(np.rint((n_levels - 1) * production / max(capacity, 1e-8)), 0, n_levels - 1))

def team_target(demand: float, rival: float, rho: float) -> float:
    return max(demand - rival, rho / (1.0 - rho) * rival)

def allocate_production(need: float, stocks: np.ndarray, capacities: np.ndarray, version: str = 'A') -> np.ndarray:
    if version == 'A':
        return np.minimum(capacities, need * capacities / capacities.sum())
    production = np.zeros_like(capacities)
    for i in np.argsort(stocks, kind='stable'):
        production[i] = min(capacities[i], need)
        need -= production[i]
    return production

def compute_market_actions(
    obs,
    obs_map,
    controller_cfg: dict,
):
    '''
    full marl obs dict gives every firm's stock (centralized controller), a single obs uses its team slice instead
    '''
    rho = controller_cfg['rho'] + controller_cfg.get('share_margin', 0.0)

    if isinstance(obs, dict):
        agent_names = [name for name in obs.keys() if 'agent' in name]
        demand = obs[agent_names[0]][obs_map['demand']][0]
        rival = obs[agent_names[0]][obs_map['target_pos']][0]
        stocks = np.array([obs[name][obs_map['self_stock']][0] for name in agent_names])
        capacities = np.array([obs[name][obs_map['self_cap']][0] for name in agent_names])
    else:
        idx = int(np.argmax(obs[obs_map['id']]))
        demand = obs[obs_map['demand']][0]
        rival = obs[obs_map['target_pos']][0]
        stocks = np.insert(obs[obs_map['team']], idx, obs[obs_map['self_stock']][0])
        capacities = np.insert(obs[obs_map['team_caps']], idx, obs[obs_map['self_cap']][0])

    need = max(0.0, team_target(demand, rival, rho) - stocks.sum())
    production = allocate_production(need, stocks, capacities, controller_cfg['version'])
    actions = [production_to_level(m, c) for m, c in zip(production, capacities)]

    if isinstance(obs, dict):
        return {name: action for name, action in zip(agent_names, actions)}
    return actions[idx]

def rival_controller(
    obs: np.ndarray,
    obs_map: dict,
    controller_cfg: dict,
) -> np.ndarray:
    '''
    assumes recieves obs['target'], returns the noiseless output of every rival in the crowd
    '''
    team_stock = obs[obs_map['rival_team_stock']]
    n_rivals = len(team_stock)

    output = (
        obs[obs_map['rival_demand']][0]
        - controller_cfg['alpha'] * obs[obs_map['rival_team_prod']][0]
        - controller_cfg['beta'] * team_stock
        - controller_cfg['gamma'] * obs[obs_map['rival_max_stock']][0]
    ) / n_rivals - controller_cfg['own_stock_beta'] * obs[obs_map['rival_own_stock']]

    return np.clip(output, 0.0, controller_cfg['capacity'] / n_rivals)
