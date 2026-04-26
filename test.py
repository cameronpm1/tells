import cv2
import scipy
import gurobipy as gp

from controllers.boat_mpc import boatMPC
from tells_environment_dynamics.test import *
from envs.make_envs import make_usv_game, make_usv_env

def discretize_matrices(state_matrix,control_matrix,mass,inertia):
    '''
    compute discretized state and control matrices using block matrix
    trick from https://en.wikipedia.org/wiki/Discretization#cite_note-2
    '''
    block = np.zeros((9,9))
    block[0:6,0:6] = state_matrix
    block[0:6,6:9] = [
        [0.0,0.0,0.0],
        [0.0,0.0,0.0],
        [1.0/mass,0.0,0.0],
        [0.0,1.0/mass,0.0],
        [0.0,0.0,0.0],
        [0.0,0.0,1.0/inertia],
    ]
    block_d = scipy.linalg.expm(block*1.0)

    A_d = block_d[0:6,0:6]
    B_d = block_d[0:6,6:9]

    return A_d,B_d

boat_points = np.array([
    [0.0, 0.25],
    [1.5, 0.25],
    [2.0, 0.0],
    [1.5, -0.25],
    [0.0, -0.25]
])

boat_lines = np.array([
    (0,1), (1,2), (2,3), (3,4), (4,0)
])

def boat_move():
    steps = 20
    boat = make_boat('boat1')

    mass = boat.get_local_attr('mass')
    inertia = boat.get_local_attr('I11')
    A_d, B_d = discretize_matrices(boat.get_local_attr('A'),boat.get_local_attr('B'),mass,inertia)

    mpc = boatMPC(
        initial_state = boat.get_local_attr('state'),
        state_matrix = A_d,
        control_matrix = B_d,
        control_bounds = np.array([200, 0, 2.0]),
        dt = 1.0,
        N = 3,
    )

    goal_state = np.array([-30,  0.0, 0.0, 0.0, 0.0, 0.0]) 

    plot_data = {}
    plot_data['lines'] = boat_lines
    plot_data['points'] = boat_points * 10

    #start plotter
    renderer = Renderer2D(xlim=[-100, 100], ylim=[-100, 100])  
    plt.ion()

    for i in range(steps):
        action = mpc.compute_action(
            initial_state = boat.get_local_attr('state'),
            reference_state = goal_state,
        )
        boat.set_ctrl([action[0],0,action[2]])
        #forward step and point computation
        boat.forward_step()
        pos = boat.get_local_attr('pos')
        dcm = boat.get_local_attr('dcm')
        transformed_vertices = np.dot(boat_points*boat.size, dcm.T) + pos
        plot_data['points'] = transformed_vertices
        #plotting
        renderer.clear()  # Clear once for all satellites
        renderer.plot(plot_data)  # Plot satellite
        plt.pause(0.05)
        #print(boat.get_local_attr('pos'),boat.get_local_attr('speed'),action[0])

    plt.ioff()
    plt.show()

def test_usv_game():
    game = make_usv_game(config_dir='/Users/cameronmehlman/Documents/tells/confs/usv_configs/game1.yaml')

    renderer = Renderer2D(xlim=[-50, 1050], ylim=[-50, 1050])  
    plt.ion()

    for j in range(1000):
        points = []
        lines = []

        if j % 1 == 0:
            target_goal = game.target_boat_PFA()
        
        action = game.set_position_control('target',target_goal)
        #forward step and point computation
        game.forward_step()

        for i,boat in enumerate(game.boats.keys()):
            pos = game.boats[boat].get_local_attr('pos')
            dcm = game.boats[boat].get_local_attr('dcm')
            transformed_vertices = np.dot(boat_points*10, dcm.T) + pos
            for j in range(len(boat_points)):
                points.append(transformed_vertices[j])
                lines.append((boat_lines[j][0]+(5*i),boat_lines[j][1]+(5*i)))

        plot_data = {}
        plot_data['lines'] = lines
        plot_data['points'] = points
        plot_data['goal'] = target_goal
        #plotting
        renderer.clear()  # Clear once for all satellites
        renderer.plot(plot_data)  # Plot satellite
        plt.pause(0.05)

    plt.ioff()
    plt.show()

def test_usv_env():
    env = make_usv_env(config_dir='/Users/cameronmehlman/Documents/tells/confs/usv_configs/game1.yaml',wrap=False)

    renderer = Renderer2D(xlim=[-50, 1050], ylim=[-50, 1050])  
    plt.ion()

    obs, _ = env.reset()

    for j in range(1000):
        action = np.zeros((16))

        _,_,_,_,_ = env.step(action)

        plot_data = env.get_plot_data()

        #plotting
        renderer.clear()  # Clear once for all satellites
        renderer.plot(plot_data)  # Plot satellite
        plt.pause(0.05)

    plt.ioff()
    plt.show()

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

if __name__ == "__main__":
    #test_boat_dynamics()
    #model = gp.Model()
    #boat_move()

    gen_belief_img()
    #test_usv_game()
    #test_usv_env()