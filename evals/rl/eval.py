import cv2
import yaml
import numpy as np

from stable_baselines3 import SAC

from envs.rl.make_envs import make_usv_env
from tells_environment_dynamics.test import *
from util.util import mkdir, load_config, save_argb_video

def eval(
        config_dir:str,
        model_dir:str,
    ):
    '''
    test policy and save gif of test run

    input
    -----
    config_dir:str
        config directory
    model_dir:str
        policy directory
    '''
    cfg = load_config(config_dir)
    env = make_usv_env(cfg,wrap=True)

    model = SAC.load(model_dir)

    renderer = Renderer2D(xlim=[-200, 1200], ylim=[-200, 1200])  
    plt.ion()

    obs, _ = env.reset()

    images = []
    total_reward = 0

    for j in range(cfg['env']['max_episode_length']):
        action, _ = model.predict(obs)
        #action = np.zeros((16))

        obs,rew,terminated,truncated,_ = env.step(action)
        total_reward += rew
        plot_data_points = _['plotting']
        
        #plotting
        for plot_data in plot_data_points:
            renderer.clear()  # Clear once for all satellites
            renderer.plot(plot_data,pause=0.0)  # Plot satellite

            image = renderer.get_rgb()

            images.append(image[:,:,1:])

        if terminated or truncated:
            break
    
    print('Episode ended in ' + str(len(images)) + ' timesteps, with total reward ' + str(total_reward))

    plt.ioff()
    plt.show()

    save_dir = 'test.mp4'
    print('generating video in ' + save_dir)
    save_argb_video(images,save_dir)

