import torch
import argparse



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--command', type=str, default='eval_rl', help='Command to run')
    parser.add_argument('--config', type=str, default=None, help='Config file path')
    parser.add_argument('--runs', type=int, default=10, help='Number of runs to test')
    parser.add_argument('--model_dir', type=str, default=None, help='Model directory for evaluation')
    parser.add_argument('--save_dir', type=str, default=None, help='Directory to save evaluation data to')
    parser.add_argument('--n_workers', type=int, default=1, help='Number of parallel workers to use for data collection')
    args = parser.parse_args()

    if args.config is None:
        print('ERROR: No config file provided')
    elif 'rl_train' in args.command:

        torch.set_num_threads(9)
        from learn.rl.train import train

        print('Training RL model with config:', args.config)
        train(args.config)
    elif 'rl_eval' in args.command:

        if args.model_dir is None:
            print('ERROR: No model directory provided for evaluation')
            exit()
        else:

            torch.set_num_threads(9)
            from evals.rl.eval import eval

            print('Evaluating RL model with config:', args.config)
            print('Loading model from:', args.model_dir)
            eval(args.config,args.model_dir)

    elif 'rl_collect_data' in args.command:

        if args.model_dir is None:
            print('ERROR: No model directory provided for evaluation')
            exit()
        else:

            torch.set_num_threads(9)
            from evals.rl.collect_eval_data import collect_data

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

    #config_dir = args.config if args.config else 'confs/usv_configs/3b_game.yaml'
    
    #train(config_dir)
    #collect_data(config_dir,'/home/cameron/tells/logs/circle_dist_noise/model.zip',master_dir='data/circle_obs',n_runs=1000)
    #test(config_dir,'/home/cameron/tells/logs/circle_dist_noise/model.zip')