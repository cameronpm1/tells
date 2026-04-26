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
        def set_diff(env):
            env.set_difficulty(new_difficulty)
            
        def update_worker(worker):
            worker.foreach_env(
                lambda env: env.set_difficulty(self.difficulty)
            )

        algorithm.env_runner_group.foreach_env_runner(update_worker)

        #tensorbaord logging
        result["curriculum/difficulty"] = self.difficulty