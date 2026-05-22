import numpy as np
from ray.rllib.algorithms.callbacks import DefaultCallbacks

class CurriculumCallback(DefaultCallbacks):

    def __init__(
        self,
    ):
        super().__init__()
        self.start_timesteps = 500000
        self.end_timesteps = 12000000 #6 was last value used
        self.start_difficulty = 0.1   # initial value
        self.max_difficulty = 1.0
        '''
        single agent params
        self.start_timesteps = 500000
        self.end_timesteps = 2500000
        self.start_difficulty = 0.1   # initial value
        self.max_difficulty = 1.0
        '''
        

    def on_train_result(self, *, algorithm, result, **kwargs):
        timesteps = result["timesteps_total"]

        if timesteps < self.start_timesteps:
            self.difficulty = self.start_difficulty
        elif timesteps < self.end_timesteps:
            self.difficulty = self.start_difficulty + (timesteps - self.start_timesteps) / (self.end_timesteps - self.start_timesteps) * (self.max_difficulty - self.start_difficulty)
        else:
            self.difficulty = self.max_difficulty

        # Push to all workers
        def update_worker(worker):
            worker.foreach_env(
                lambda env: env.set_difficulty(self.difficulty)
            )

        algorithm.env_runner_group.foreach_env_runner(update_worker)

        #tensorbaord logging
        result["curriculum/difficulty"] = self.difficulty

class LogRawEpisodeReturn(DefaultCallbacks):
    @staticmethod
    def _update_metric(metric_store, key, value):
        entry = metric_store.setdefault(
            key,
            {
                'sum': 0.0,
                'count': 0,
                'min': float('inf'),
                'max': float('-inf'),
                'last': 0.0,
            },
        )
        value = float(value)
        entry['sum'] += value
        entry['count'] += 1
        entry['min'] = min(entry['min'], value)
        entry['max'] = max(entry['max'], value)
        entry['last'] = value

    def on_episode_start(
        self,
        *,
        worker,
        base_env,
        policies,
        episode,
        env_index,
        **kwargs,
    ):

        episode.user_data['raw_return'] = 0.0
        episode.user_data['env_metrics'] = {}

    def on_episode_step(
        self,
        *,
        worker,
        base_env,
        episode,
        env_index,
        **kwargs,
    ):

        env = base_env.get_sub_environments()[env_index]

        if hasattr(env, 'last_raw_reward'):
            episode.user_data['raw_return'] += float(env.last_raw_reward)
        source_env = getattr(env, 'env', env)
        metrics = getattr(source_env, 'last_metrics', None)
        if isinstance(metrics, dict):
            metric_store = episode.user_data.setdefault('env_metrics', {})
            for key, value in metrics.items():
                if isinstance(value, (bool, int, float, np.bool_, np.integer, np.floating)):
                    self._update_metric(metric_store, key, value)

    def on_episode_end(
        self,
        *,
        worker,
        base_env,
        policies,
        episode,
        env_index,
        **kwargs,
    ):
        raw_return = float(episode.user_data.get('raw_return', 0.0))

        episode.custom_metrics['raw_episode_return'] = raw_return
        metric_store = episode.user_data.get('env_metrics', {})
        if not metric_store:
            return

        if 'target_goal_dist' in metric_store:
            stats = metric_store['target_goal_dist']
            episode.custom_metrics['target_goal_dist_min'] = float(stats['min'])
            episode.custom_metrics['target_goal_dist_final'] = float(stats['last'])
        if 'hold_steps' in metric_store:
            episode.custom_metrics['hold_steps_max'] = float(metric_store['hold_steps']['max'])
        if 'success' in metric_store:
            episode.custom_metrics['success'] = float(metric_store['success']['max'])
        if 'oob' in metric_store:
            episode.custom_metrics['oob'] = float(metric_store['oob']['max'])
        for key in (
            'ring_score',
            'coverage_score',
            'slot_score',
            'slot_distance',
            'close_fraction',
            'control_score',
            'goal_center_score',
            'goal_lock_score',
            'touch_penalty',
        ):
            if key in metric_store:
                stats = metric_store[key]
                episode.custom_metrics[key + '_mean'] = float(stats['sum'] / max(stats['count'], 1))
