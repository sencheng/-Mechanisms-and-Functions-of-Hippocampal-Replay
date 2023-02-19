# basic imports
import os
import numpy as np
import pickle
import pyqtgraph as qg
# framework imports
from cobel.agents.sfma import SFMAAgent
from cobel.interfaces.gridworld import InterfaceGridworld
from cobel.misc.gridworld_tools import make_gridworld
# store experiment directory
cwd = os.getcwd()

# shall the system provide visual output while performing the experiments?
# NOTE: do NOT use visualOutput=True in parallel experiments, visualOutput=True should only be used in explicit calls to 'singleRun'! 
visual_output = False


def single_run(attention_arms: float, attention_cued: float, gamma=0.8, mode='default') -> (list, np.ndarray):
    '''
    This method performs a single experimental run, i.e. one experiment.
    It has to be called by either a parallelization mechanism (without visual output),
    or by a direct call (in this case, visual output can be used).
    
    Parameters
    ----------
    attention_arms :                    Modulates strenght increase due to visual exploration.
    attention_cued :                    Fraction of attention paid to either arm.
    gamma :                             The Default Representation's discount factor.
    mode :                              The replay mode that will be used.
    
    Returns
    ----------
    replays :                           The list of generated preplays and replays.
    C :                                 The experience strengths.
    '''
    np.random.seed()
    # this is the main window for visual output
    # normally, there is no visual output, so there is no need for an output window
    main_window = None
    # if visual output is required, activate an output window
    if visual_output:
        main_window = qg.GraphicsWindow(title='Simulation: Preplay')
    
    # define walls
    invalid_transitions = []
    invalid_transitions += [(0, 7), (1, 8), (2, 9), (4, 11), (5, 12), (6, 13), (17, 24)]
    invalid_transitions += [(7, 0), (8, 1), (9, 2), (11, 4), (12, 5), (13, 6), (24, 17)]
    invalid_transitions += [(9, 10), (16, 17), (23, 24), (30, 31), (37, 38), (44, 45), (51, 52), (58, 59), (65, 66)]
    invalid_transitions += [(10, 9), (17, 16), (24, 23), (31, 30), (38, 37), (45, 44), (52, 51), (59, 58), (66, 65)]
    invalid_transitions += [(10, 11), (17, 18), (24, 25), (31, 32), (38, 39), (45, 46), (52, 53), (59, 60), (66, 67)]
    invalid_transitions += [(11, 10), (18, 17), (25, 24), (32, 31), (39, 38), (46, 45), (53, 52), (60, 59), (67, 66)]
    
    # initialize world            
    world = make_gridworld(10, 7, terminals=[0, 6], rewards=np.array([[6, 1]]), goals=[6], starting_states=[63], invalid_transitions=invalid_transitions)    
    
    # a dictionary that contains all employed modules
    modules = {}
    modules['rl_interface'] = InterfaceGridworld(modules, world, visual_output, main_window)
    
    # initialize RL agent
    rl_agent = SFMAAgent(modules['rl_interface'], 0.3, 5, 0.9, 0.9, gamma_SR=gamma)
    rl_agent.M.decay_inhibition = 0.9
    rl_agent.M.beta = 9
    rl_agent.M.C_normalize = False
    rl_agent.M.D_normalize = False
    rl_agent.M.R_normalize = True
    rl_agent.M.mode = mode
    
    # maze parts
    cued = np.array([3 + 140, 4 + 140, 5 + 140, 6 + 140, 10 + 70, 17 + 70, 3 + 210, 4, 5, 6, 10 + 210, 17 + 210])
    uncued = np.array([0, 1, 2, 3, 10 + 70, 17 + 70, 0 + 140, 1 + 140, 2 + 140, 3 + 140, 10 + 210, 17 + 210])
    stem = np.array([24, 31, 38, 45, 52, 59, 66])
    
    # predefine behavioral patterns
    predefined_C = np.zeros(280)
    # stem (running up & down the stem)
    predefined_C[stem + 70] = 10.
    predefined_C[stem + 210] = 10.
    predefined_C +=  10 * np.tile(rl_agent.M.metric.D[24] + rl_agent.M.metric.D[66], 4)
    # arms (attention being paid to the arms and reward cue)
    # visual exploration
    predefined_C[cued] += attention_cued * 10 * attention_arms
    predefined_C[uncued] += (1. - attention_cued) * 10 * attention_arms
    # visual reward modulation
    predefined_C += np.tile(rl_agent.M.metric.D[6], 4) * 10 * attention_arms * attention_cued
    rl_agent.M.C = predefined_C
    
    # initialize experiences
    for state in range(world['sas'].shape[0]):
        for action in range(world['sas'].shape[1]):
            rl_agent.M.states[state][action] = np.argmax(world['sas'][state][action])
    
    # start replays
    replays = []
    for i in range(5000):
        replays += [rl_agent.M.replay(6)]
    
    # and also stop visualization
    if visual_output:
        main_window.close()
        
    return replays, rl_agent.M.C

if __name__ == '__main__':
    # params
    attention_cued = np.array([0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0])
    attention_arms = np.array([0.0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2])
    modes = ['default', 'reverse']
    gammas = [0.1, 0.8]
    
    # start simulations
    for gamma in gammas:
        for mode in modes:
            print('Discount factor: ' + str(gamma) + ', Mode: ' + mode)
            for cued in attention_cued:
                for arms in attention_arms:
                    print('Starting replays. Attention Cued: ' + str(cued) + ', Attention Arms: ' + str(arms))
                    replays, C = single_run(arms, cued, gamma=gamma, mode=mode)
                    file_name = cwd + '/data/sweep/gamma_' + str(gamma) + '_mode_' + mode + '_arms_' + str(arms) + '_cued_' + str(cued) + '.pkl'
                    pickle.dump(replays, open(file_name, 'wb'))
