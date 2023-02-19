# basic imports
import os
import gc
import pickle
import numpy as np
import pyqtgraph as qg
# framework imports
from cobel.agents.sfma import SFMAAgent
from cobel.interfaces.gridworld import InterfaceGridworld
from cobel.misc.gridworld_tools import make_empty_field
from cobel.memory_modules.memory_utils.metrics import Euclidean
# store experiment directory
cwd = os.getcwd()

# shall the system provide visual output while performing the experiments?
# NOTE: do NOT use visualOutput=True in parallel experiments, visualOutput=True should only be used in explicit calls to 'singleRun'! 
visual_output = False


def single_run(euclidean_similarity=False, mode='default') -> dict:
    '''
    This method performs a single experimental run, i.e. one experiment.
    It has to be called by either a parallelization mechanism (without visual output),
    or by a direct call (in this case, visual output can be used).
    
    Parameters
    ----------
    euclidean_similarity :              If true, experience similarity will be based on the euclidean distance between states.
    mode :                              The replay mode that will be used.
    
    Returns
    ----------
    replays :                           Dictionary containing the replays generated in each environment.
    '''
    np.random.seed()
    # this is the main window for visual output
    # normally, there is no visual output, so there is no need for an output window
    main_window = None
    # if visual output is required, activate an output window
    if visual_output:
        main_window = qg.GraphicsWindow(title="workingTitle_Framework")
    
    # initialize world            
    world = make_empty_field(10, 10)
    
    # a dictionary that contains all employed modules
    modules = {}
    modules['rl_interface'] = InterfaceGridworld(modules, world, visual_output, main_window)
    
    # initialize RL agent
    rl_agent = SFMAAgent(interface_OAI=modules['rl_interface'], epsilon=0.3, beta=5,
                       learning_rate=0.9, gamma=0.9, gamma_SR=0.1)
    rl_agent.M.C.fill(1)
    rl_agent.M.decay_inhibition = 0.9
    rl_agent.M.beta = 9
    rl_agent.M.C_normalize = False
    rl_agent.M.D_normalize = False
    rl_agent.M.R_normalize = True
    rl_agent.M.mode = mode
    if euclidean_similarity:
        rl_agent.M.metric = Euclidean(modules['rl_interface'])
        rl_agent.M.mode = 'default'
    
    # eventually, allow the OAI class to access the robotic agent class
    modules['rl_interface'].rl_agent = rl_agent
    
    # and also stop visualization
    if visual_output:
        main_window.close()
        
    replays = {'env_1': [], 'env_2': [], 'env_3': []}
        
    # replay in first environment
    print('Start Replays - Environment 1.')
    replays['env_1'] = []
    for i in range(100):
        if (i+1) % 10 == 0:
            print('\tReplay: ', str(i+1))
        replays['env_1'] += [rl_agent.M.replay(50, 55)]
    
    # change environment
    invalid_transitions = [(1, 2), (11, 12), (21, 22), (31, 32), (41, 42), (51, 52), (61, 62), (71, 72)]
    invalid_transitions += [(2, 1), (12, 11), (22, 21), (32, 31), (42, 41), (52, 51), (62, 61), (72, 71)]
    invalid_transitions += [(14, 24), (15, 25), (16, 26), (17, 27)]
    invalid_transitions += [(24, 14), (25, 15), (26, 16), (27, 17)]
    invalid_transitions += [(23, 24), (33, 34), (43, 44), (53, 54), (63, 64), (73, 74), (83, 84), (93, 94)]
    invalid_transitions += [(24, 23), (34, 33), (44, 43), (54, 53), (64, 63), (74, 73), (84, 83), (94, 93)]
    invalid_transitions += [(36, 46), (37, 47), (38, 48), (39, 49)]
    invalid_transitions += [(46, 36), (47, 37), (48, 38), (49, 39)]
    invalid_transitions += [(76, 86), (77, 87), (86, 76), (87, 77)]
    invalid_transitions += [(45, 46), (55, 56), (65, 66), (75, 76)]
    invalid_transitions += [(46, 45), (56, 55), (66, 65), (76, 75)]
    modules['rl_interface'].update_transitions(invalid_transitions)
    rl_agent.M.metric.update_transitions()
    # replay in second environment
    print('Start Replays - Environment 2.')
    replays['env_2'] = []
    for i in range(100):
        if (i+1) % 10 == 0:
            print('\tReplay: ', str(i+1))
        replays['env_2'] += [rl_agent.M.replay(50, 55)]
    
    # change environment
    invalid_transitions = [(4, 5), (14, 15), (24, 25), (34, 35), (64, 65), (74, 75), (84, 85), (94, 95)]
    invalid_transitions += [(5, 4), (15, 14), (25, 24), (35, 34), (65, 64), (75, 74), (85, 84), (95, 94)]
    invalid_transitions += [(22, 23), (32, 33), (26, 27), (36, 37)]
    invalid_transitions += [(23, 22), (33, 32), (27, 26), (37, 36)]
    invalid_transitions += [(62, 63), (72, 73), (66, 67), (76, 77)]
    invalid_transitions += [(63, 62), (73, 72), (67, 66), (77, 76)]
    invalid_transitions += [(33, 43), (34, 44), (35, 45), (36, 46), (53, 63), (54, 64), (55, 65), (56, 66)]
    invalid_transitions += [(43, 33), (44, 34), (45, 35), (46, 36), (63, 53), (64, 54), (65, 55), (66, 56)]
    modules['rl_interface'].update_transitions(invalid_transitions)
    rl_agent.M.metric.update_transitions()
    # replay in third environment
    print('Start Replays - Environment 3.')
    replays['env_3'] = []
    for i in range(100):
        if (i+1) % 10 == 0:
            print('\tReplay: ', str(i+1))
        replays['env_3'] += [rl_agent.M.replay(50, 55)]
        
    return replays


if __name__ == '__main__':
    # run simulations with DR similarity
    replays_DR = single_run()
    pickle.dump(replays_DR, open(cwd + '/data/replays_DR_default.pkl', 'wb'))
    replays_DR = single_run(mode='reverse')
    pickle.dump(replays_DR, open(cwd + '/data/replays_DR_reverse.pkl', 'wb'))
    # run simulations with Euclidean similarity
    replays_euclidean = single_run(True)
    pickle.dump(replays_euclidean, open(cwd + '/data/replays_euclidean.pkl', 'wb'))
    gc.collect()
