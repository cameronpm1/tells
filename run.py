import os
import ray
import torch
import argparse
import subprocess
from pathlib import Path

def set_torch_threads(default: int = 9):
    torch.set_num_threads(int(os.environ.get('TELL_TORCH_THREADS', default)))

def ensure_pybullet_drones_source():
    repo_root = Path(__file__).resolve().parent
    submodule_dir = repo_root / 'external' / 'pybullet-drones'
    package_dir = submodule_dir / 'gym_pybullet_drones'
    pinned_commit = '90b178aba69b09085dd70e7a1e88bb58ca00b4c9'

    if package_dir.exists():
        return

    if (repo_root / '.git').exists():
        subprocess.run(
            ['git', 'submodule', 'update', '--init', '--recursive', 'external/pybullet-drones'],
            cwd=repo_root,
            check=True,
        )
        if package_dir.exists():
            return

    if submodule_dir.exists() and not any(submodule_dir.iterdir()):
        submodule_dir.rmdir()
    if not submodule_dir.exists():
        submodule_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ['git', 'clone', 'https://github.com/utiasDSL/gym-pybullet-drones.git', str(submodule_dir)],
            cwd=repo_root,
            check=True,
        )
        subprocess.run(
            ['git', '-C', str(submodule_dir), 'checkout', pinned_commit],
            cwd=repo_root,
            check=True,
        )

    if not package_dir.exists():
        raise RuntimeError(
            'Could not find gym_pybullet_drones. Run '
            '`git submodule update --init --recursive external/pybullet-drones` '
            'or allow this script to clone the submodule fallback.'
        )

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--command', type=str, default='eval_rl', help='Command to run')
    parser.add_argument('--config', type=str, default=None, help='Config file path')
    parser.add_argument('--runs', type=int, default=10, help='Number of runs to test')
    parser.add_argument('--model_dir', type=str, default=None, help='Model directory for evaluation')
    parser.add_argument('--belief_config', type=str, default=None, help='Config file for belief model')
    parser.add_argument('--belief_dir', type=str, default=None, help='Model directory for belief model')
    parser.add_argument('--save_dir', type=str, default=None, help='Directory to save evaluation data to')
    parser.add_argument('--n_workers', type=int, default=1, help='Number of parallel workers to use for data collection')
    args = parser.parse_args()

    if args.config is None:
        print('ERROR: No config file provided')
    elif args.command == 'rl_train':

        set_torch_threads()
        from learn.rl.train import train

        print('Training RL model with config:', args.config)
        train(args.config)
    elif args.command == 'rl_eval':

        if args.model_dir is None:
            print('ERROR: No model directory provided for evaluation')
            exit()
        else:

            set_torch_threads()
            from evals.rl.eval import eval

            print('Evaluating RL model with config:', args.config)
            print('Loading model from:', args.model_dir)
            eval(args.config,args.model_dir)

    elif 'rl_collect_data' == args.command:

        if args.model_dir is None:
            print('ERROR: No model directory provided for evaluation')
            exit()
        else:

            set_torch_threads()
            from evals.rl.collect_eval_data import collect_data

            if args.save_dir is None:
                save_dir = 'data/' + args.model_dir.split('rl/')[1].split('/')[0]
            else:
                save_dir = 'data/' + args.save_dir
            print('Collecting data with RL model and config:', args.config)
            print('Loading model from:', args.model_dir)
            print('Saving data to:', save_dir)

            collect_data(args.config, args.model_dir, save_dir, n_runs=args.runs, n_workers=args.n_workers)

    elif 'marl_train' in args.command:

        
        import warnings
        os.environ["XDG_RUNTIME_DIR"] = "/tmp"
        os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=UserWarning, module="pygame")
        set_torch_threads()
        ensure_pybullet_drones_source()

        from learn.marl.train import train

        ray_kwargs = {
            'runtime_env': {
                'working_dir': os.getcwd(),
                'env_vars': {'PYTHONWARNINGS': 'ignore::DeprecationWarning'},
                'excludes': ['.git/', '.venv/', 'logs/', 'lab-logs/', 'uv.lock', '__pycache__/', '**/__pycache__/', '*.pyc', '.DS_Store'],
            },
        }
        if os.environ.get('RAY_TMPDIR'):
            ray_kwargs['_temp_dir'] = os.environ['RAY_TMPDIR']
        ray.init(**ray_kwargs)

        print('Training marl policies with config:', args.config)
        train(args.config)

    elif  args.command == 'marl_eval':

        
        import warnings
        os.environ["XDG_RUNTIME_DIR"] = "/tmp"
        os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=UserWarning, module="pygame")
        set_torch_threads()
        ensure_pybullet_drones_source()

        from evals.marl.eval import eval

        ray_kwargs = {
            'runtime_env': {
                'working_dir': os.getcwd(),
                'env_vars': {'PYTHONWARNINGS': 'ignore::DeprecationWarning'},
                'excludes': ['.git/', '.venv/', 'logs/', 'lab-logs/', 'uv.lock', '__pycache__/', '**/__pycache__/', '*.pyc', '.DS_Store'],
            },
        }
        if os.environ.get('RAY_TMPDIR'):
            ray_kwargs['_temp_dir'] = os.environ['RAY_TMPDIR']
        ray.init(**ray_kwargs)

        print('Evaluating RL model with config:', args.config)
        print('Loading model from:', args.model_dir)
        eval(args.config,args.model_dir,args.runs)

    elif args.command == 'marl_eval_belief':

        
        import warnings
        os.environ["XDG_RUNTIME_DIR"] = "/tmp"
        os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=UserWarning, module="pygame")
        set_torch_threads()
        ensure_pybullet_drones_source()

        from evals.marl.eval import eval

        ray.init(runtime_env={'working_dir': os.getcwd(),
                              'env_vars': {'PYTHONWARNINGS': 'ignore::DeprecationWarning'},
                              'excludes': ['.git/', '.venv/', 'logs/', 'lab-logs/', 'uv.lock', '__pycache__/', '**/__pycache__/', '*.pyc', '.DS_Store']
                              } )

        print('Evaluating RL model with config:', args.config)
        print('Loading model from:', args.model_dir)
        eval(args.config,args.model_dir,args.runs,args.belief_dir,args.belief_config)

    elif 'marl_collect_data' == args.command:

        if args.model_dir is None:
            print('ERROR: No model directory provided for evaluation')
            exit()
        else:

            set_torch_threads()
            from evals.marl.collect_eval_data import collect_data

            if args.save_dir is None:
                save_dir = 'data/' + args.model_dir.split('rl/')[1].split('/')[0]
            else:
                save_dir = 'data/' + args.save_dir
            print('Collecting data with RL model and config:', args.config)
            print('Loading model from:', args.model_dir)
            print('Saving data to:', save_dir)
            
            collect_data(args.config, args.model_dir, save_dir, n_runs=args.runs, n_workers=args.n_workers)

    elif 'belief_train' in args.command:

        from learn.belief.train import train

        print('Training belief model with config:', args.config)
        train(args.config)

    elif 'belief_eval' in args.command:

        if args.model_dir is None:
            print('ERROR: No model directory provided for evaluation')
            exit()
        else:
        
            from evals.belief.eval import eval
            
            print('Evaluating belief model with config:', args.config)
            print('Loading model from:', args.model_dir)

            eval(args.config,args.model_dir)

    elif 'ic3_train' in args.command:

        from learn.marl.train_IC3Net import train

        print('Training IC3Net model with config:', args.config)
        train(args.config)

    elif 'ic3_eval' in args.command:

        from evals.marl.eval_IC3Net import eval

        print('Evaluating IC3Net model with config:', args.config)
        print('Loading model from:', args.model_dir)
        eval(args.config,args.model_dir,args.runs)

    elif 'controller_eval' in args.command:

        from learn.marl.train import test_controller

        print('Evaluating controller with config:', args.config)
        test_controller(args.config, args.runs)

    #config_dir = args.config if args.config else 'confs/usv_configs/3b_game.yaml'
    
    #train(config_dir)
    #collect_data(config_dir,'/home/cameron/tells/logs/circle_dist_noise/model.zip',master_dir='data/circle_obs',n_runs=1000)
    #test(config_dir,'/home/cameron/tells/logs/circle_dist_noise/model.zip')
