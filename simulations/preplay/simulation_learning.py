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


def single_run(fraction, stem_only=False, proximity=False, reward_modulation=0.1, attention=1., r=1., gamma=0.8) -> (np.ndarray, np.ndarray):
    '''
    This method performs a single experimental run, i.e. one experiment.
    It has to be called by either a parallelization mechanism (without visual output),
    or by a direct call (in this case, visual output can be used).
    
    Parameters
    ----------
    fraction :                          Modulates strenght increase due to visual exploration.
    stem_only :                         Fraction of attention paid to either arm.
    proximity :                         The Default Representation's discount factor.
    reward_modulation :                 The replay mode that will be used.
    attention :                         The replay mode that will be used.
    r :                                 The replay mode that will be used.
    gamma :                             The discount factor used for computing the DR.
    
    Returns
    ----------
    Q_before :                          The agent's Q function before preplay.
    Q_after :                           The agent's Q function after preplay.
    '''
    np.random.seed()
    # this is the main window for visual output
    # normally, there is no visual output, so there is no need for an output window
    main_window = None
    # if visual output is required, activate an output window
    if visual_output:
        main_window = qg.GraphicsWindow(title='Simulation: Preplay and Learning')
    
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
    rl_agent.gamma = 0.99
    rl_agent.M.decay_inhibition = 0.9
    rl_agent.M.beta = 9
    rl_agent.M.C_normalize = False
    rl_agent.M.D_normalize = False
    rl_agent.M.R_normalize = True
    
    # initialize experience
    for state in range(70):
        for action in range(4):
            rl_agent.M.states[state, action] = np.argmax(world['sas'][state, action])
    rl_agent.M.rewards[5, 2] = 1.
    rl_agent.M.terminals.fill(1)
    rl_agent.M.terminals[5, 2] = 0
    rl_agent.M.terminals[6, :] = 0
    
    # maze parts
    cued = np.array([3, 4, 5, 6, 10, 17])
    uncued = np.array([0, 1, 2, 3, 10, 17])
    stem = np.array([24, 31, 38, 45, 52, 59, 66])
    
    # maze parts
    cued = np.array([3 + 140, 4 + 140, 5 + 140, 6 + 140, 10 + 70, 17 + 70, 3 + 210, 4, 5, 6, 10 + 210, 17 + 210])
    uncued = np.array([0, 1, 2, 3, 10 + 70, 17 + 70, 0 + 140, 1 + 140, 2 + 140, 3 + 140, 10 + 210, 17 + 210])
    stem = np.array([24, 31, 38, 45, 52, 59, 66])
    
    # predefine experience strengths
    predefined_C = np.zeros(280)
    # stem
    predefined_C[stem + 70] = 10.
    predefined_C[stem + 210] = 10.
    predefined_C += 10 * np.tile(rl_agent.M.metric.D[24] + rl_agent.M.metric.D[66], 4) * r
    # arms
    if not stem_only:
        predefined_C[cued] += fraction * 10 * attention
        predefined_C[uncued] += (1. - fraction) * 10 * attention
        if proximity:
            predefined_C += reward_modulation * np.tile(rl_agent.M.metric.D[17], 4) * 16
            #predefined_C *= reward_modulation * np.tile(rl_agent.M.metric.D[17] + 1., 4)
        predefined_C += reward_modulation * np.tile(rl_agent.M.metric.D[6], 4) * 20 * fraction
    rl_agent.M.C = predefined_C
    
    # start replays
    replays = []
    for i in range(500):
        replays += [rl_agent.M.replay(6)]
    
    # store initial Q
    Q_before = np.copy(rl_agent.Q)
    
    # train on replays
    for replay in replays:
        for experience in replay:
            rl_agent.update_Q(experience)
    
    # and also stop visualization
    if visual_output:
        main_window.close()
        
    return Q_before, np.copy(rl_agent.Q)

if __name__ == '__main__':
    # params
    modes = ['default', 'reverse']
    gammas = [0.1, 0.8]
    target = 7.37
    
    # start simulations
    for gamma in gammas:
        for mode in modes:
            print('Discount factor: ' + str(gamma) + ', Mode: ' + mode)
            # load sweep data
            sweep = pickle.load(open(cwd + '/data/analyzed_preplay_gamma_' + str(gamma) + '_mode_' + mode + '.pkl', 'rb'))
            # determine parameters which best reproduce the fraction of preplays reported by Olafsdottir et al. (2015)
            fractions = np.abs(sweep['preplay'] - target)
            best = np.argmin(fractions)
            best_fraction, best_r = np.unravel_index(best, (11, 11))
            best_fraction, best_r = sweep['fractions'][best_fraction], sweep['R'][best_r]
            # Preplay & Train
            Q_before, Q_after = single_run(best_fraction, reward_modulation=0., proximity=False, r=best_r)
            pickle.dump({'before': Q_before, 'after': Q_after}, open(cwd + '/data/analyzed_learning_gamma_' + str(gamma) + '_mode_' + mode + '.pkl', 'wb'))
