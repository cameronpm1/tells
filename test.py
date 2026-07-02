import cv2
import time
import numpy as np

from util.util import mkdir, load_config
from envs.marl.football_env import CirclePass5v1Env
from envs.marl.drones_env import PredatorPreyAviary
from controllers.football_control import compute_rondo_actions
from util.util import mkdir, load_config, save_argb_video, save_rgb_gif 
from envs.marl.make_env import make_drones_env, make_predator_prey_env, make_marl_env

#from gym_pybullet_drones.utils.enums import ActionType

import gfootball.env as football_env


def gen_belief_img():

    datapoint = np.load('/home/cameron/tells/data/circle_obs/2/step_4.npz', allow_pickle=True)
    data_labels = datapoint.files

    data = datapoint['target_true'] 
    label = np.zeros((100,100))

    for i in range(len(data_labels) - 1):
        idx = next((j for j,label in enumerate(data_labels) if str(i) in label), -1)
        loc = (datapoint[data_labels[idx]][0:2] + np.array([0.5,0.5]))*100
        label[np.clip(int(loc[1]),0,99),np.clip(int(loc[0]),0,99)] = 1

    grayscale_image = cv2.convertScaleAbs(label, alpha=255.0)
    cv2.imwrite("output_grayscale.png", grayscale_image)

def drone_test():

    config_path = '/home/cameron/tells/confs/drones/3a_game.yaml'
    cfg = load_config(config_path)


    env = make_marl_env(config=cfg,wrap='rllib')

    obs, info = env.reset(seed=1)
    images = []

    for _ in range(cfg['env']['max_episode_length']):  # 60 sec at ctrl_freq=30
        # For ActionType.VEL, action shape is:
        # (num_pursuers, 4) = [vx_dir, vy_dir, vz_dir, speed_fraction]
        # All zeros means commanded hover/no motion.
        action = {}
        for agent in env.agents:
            action[agent] = 0

        obs, reward, terminated, truncated, info = env.step(action)

        time.sleep(0.1)
        
        if terminated['__all__'] or truncated['__all__']:
            break

        images.append(env.render_rgb())



    save_file = 'test_vid.gif'
    print('generating video in ' + save_file)
    save_rgb_gif(images,save_file)

    env.close()

def football_test():
    config_path = '/home/cameron/tells/confs/football/5a_game.yaml'
    cfg = load_config(config_path)

    env = make_marl_env(config=cfg,wrap='rllib')

    obs = env.reset()
    images1 = []
    images2 = []

    max_episode_length = 900

    for step in range(max_episode_length):
        #actions = np.random.randint(0, 10, size=5)
        actions = compute_rondo_actions(obs, env.env.obs_map)
        obs, reward, term, trunc, info = env.step(actions)

        if term['__all__'] or trunc['__all__']:
            break

        frame1 = env.env.render_rgb()
        #frame2 = env.env.render_rgb_old()
        images1.append(frame1)
        #images2.append(frame2)
    obs = env.reset()
    print('!!!!!!!!!!!!!!!!!!')
    for step in range(max_episode_length):
        actions = np.random.randint(0, 10, size=5)
        obs, reward, term, trunc, info = env.step(actions)
        if term['__all__'] or trunc['__all__']:
            break
        frame1 = env.env.render_rgb()
        images1.append(frame1)
    
        '''
        print(
            f"step={step + 1}, "
            f"reward={np.asarray(reward)}, "
            f"successful_passes={info.get('successful_passes')}, "
            f"ball_owned_team={info.get('ball_owned_team')}, "
            f"ball_owned_player={info.get('ball_owned_player')}, "
            f"done={done}",
            flush=True,
        )
        '''

    save_file = "test_vid1.gif"
    print("generating video in " + save_file)   
    save_rgb_gif(images1, save_file)

    #save_file = "test_vid2.gif"
    #print("generating video in " + save_file)   
    #save_rgb_gif(images2, save_file)

    env.close()

def fire_test():
    config_path = '/home/cameron/tells/confs/fire/4a_game.yaml'
    cfg = load_config(config_path)

    env = make_marl_env(config=cfg,wrap='rllib')

    obs = env.reset()
    images = []

    max_episode_length = 20

    for step in range(max_episode_length):
        actions = {'agent0': 0, 'agent1': 0, 'agent2': 0, 'agent3': 0}
        obs, reward, term, trunc, info = env.step(actions)

        if term['__all__'] or trunc['__all__']:
            break

        frame = env.env.render_rgb()
        images.append(frame)

    save_file = "test_vid.gif"
    print("generating video in " + save_file)   
    save_rgb_gif(images, save_file)

    #save_file = "test_vid2.gif"
    #print("generating video in " + save_file)   
    #save_rgb_gif(images2, save_file)

    env.close()


if __name__ == "__main__":
    #test_boat_dynamics()
    #model = gp.Model()
    #boat_move()

    #gen_belief_img()
    #test_usv_game()
    #test_usv_env()
    #drone_test()
    football_test()
    #fire_test()