# basic imports
import os
import pickle
import numpy as np
import pyqtgraph as qg
# framework imports
from cobel.agents.sfma import SFMAAgent
from cobel.interfaces.gridworld import InterfaceGridworld
from cobel.misc.gridworld_tools import make_gridworld



def define_run_patterns() -> dict:
    '''
    This function predefines run patterns for the agent (i.e. the chosen transition).
    The run patterns are defined so that they can used to reproduce the experimental
    conditions as described by Gupta et al. (2010).
    
    Parameters
    ----------
    None
    
    Returns
    ----------
    run_patterns :                      A dictionary containing the predefined run patterns.
    '''
    run_patterns = {}
    # start to left (used to reproduce alternating laps and left laps conditions)
    run_patterns['start_to_left'] = np.array([148, 137, 126, 115, 104, 93, 5, 4, 3, 2, 1, 231, 242, 253])
    # start to right (used to reproduce alternating laps and right laps condition)
    run_patterns['start_to_right'] = np.array([148, 137, 126, 115, 104, 93, 159, 160, 161, 162, 163, 241, 252, 263])
    # left arm to left arm (used to reproduce left laps condition)
    run_patterns['left_to_left'] = np.array([264, 275, 286, 220, 221, 222, 223, 224, 148, 137, 126, 115, 104, 93, 5, 4, 3, 2, 1, 231, 242, 253])
    # left arm to right arm (used to reproduce alternating laps condition)
    run_patterns['left_to_right'] = np.array([264, 275, 286, 220, 221, 222, 223, 224, 148, 137, 126, 115, 104, 93, 159, 160, 161, 162, 163, 241, 252, 263])
    # right arm to right arm (used to reproduce right laps condition)
    run_patterns['right_to_right'] = np.array([274, 285, 296, 76, 75, 74, 73, 72, 148, 137, 126, 115, 104, 93, 159, 160, 161, 162, 163, 241, 252, 263])
    # right arm to left arm (used to reproduce alternating laps condition)
    run_patterns['right_to_left'] = np.array([274, 285, 296, 76, 75, 74, 73, 72, 148, 137, 126, 115, 104, 93, 5, 4, 3, 2, 1, 231, 242, 253])
    
    return run_patterns

def define_conditions(block_length: int = 10) -> dict:
    '''
    This function defines the different experimental conditions as described by Gupta et al. (2010).
    
    Parameters
    ----------
    block_length :                      The number of trials a condition block lasts.
    
    Returns
    ----------
    conditions :                        A dictionary containing the different experimental conditions.
    '''
    # ensure valid block length
    if block_length == 1:
        block_length = 2
    elif block_length % 2 != 0:
        block_length += 1
    # define conditions
    conditions = {}
    # static conditions
    conditions['LL'] = ['start_to_left'] + ['left_to_left' for i in range(int(block_length * 2))]
    conditions['RR'] = ['start_to_right'] + ['right_to_right' for i in range(int(block_length * 2))]
    conditions['AA'] = ['start_to_right']
    for i in range(block_length):
        conditions['AA'] += ['right_to_left', 'left_to_right']
    # changing conditions
    conditions['LR'] = ['start_to_left'] + ['left_to_left' for i in range(block_length)] + ['left_to_right'] + ['right_to_right' for i in range(block_length - 1)]
    conditions['RL'] = ['start_to_right'] + ['right_to_right' for i in range(block_length)] + ['right_to_left'] + ['left_to_left' for i in range(block_length - 1)]
    conditions['LA'] = ['start_to_left'] + ['left_to_left' for i in range(block_length)]
    conditions['RA'] = ['start_to_right'] + ['right_to_right' for i in range(block_length)]
    conditions['AL'] = ['start_to_right']
    conditions['AR'] = ['start_to_right']
    for i in range(int(block_length/2)):
        conditions['LA'] += ['left_to_right', 'right_to_left']
        conditions['RA'] += ['right_to_left', 'left_to_right']
        conditions['AL'] += ['right_to_left', 'left_to_right']
        conditions['AR'] += ['right_to_left', 'left_to_right']
    conditions['AL'] += ['right_to_left'] + ['left_to_left' for i in range(block_length - 1)]
    conditions['AR'] += ['right_to_right' for i in range(block_length)]
    
    return conditions

def single_run(run_patterns: dict, conditions: dict) -> dict:
    '''
    This function simulates a virutal version of the experimental paradigm by Gupta et al. (2010).
    Simulations are repeated for different model parameters (i.e. temperature, replay mode and effects of recency).
    The discount factor is kept fixed.
    
    Parameters
    ----------
    run_patterns :                      The set of predefined run patterns.
    conditions :                        The experimental conditions represented as lists of run patterns.
    
    Returns
    ----------
    strengths :                         A dictionary containing experiences strengths of the different experimental conditions.
    '''
    np.random.seed()
    # define walls
    invalid_transitions = []
    invalid_transitions += [(11, 12), (22, 23), (33, 34), (44, 45), (55, 56)]
    invalid_transitions += [(12, 11), (23, 22), (34, 33), (45, 44), (56, 55)]
    invalid_transitions += [(15, 16), (26, 27), (37, 38), (48, 49), (59, 60)]
    invalid_transitions += [(16, 15), (27, 26), (38, 37), (49, 48), (60, 59)]
    invalid_transitions += [(16, 17), (27, 28), (38, 39), (49, 50), (60, 61)]
    invalid_transitions += [(17, 16), (28, 27), (39, 38), (50, 49), (61, 60)]
    invalid_transitions += [(20, 21), (31, 32), (42, 43), (53, 54), (64, 65)]
    invalid_transitions += [(21, 20), (32, 31), (43, 42), (54, 53), (65, 64)]
    invalid_transitions += [(1, 12), (2, 13), (3, 14), (4, 15), (6, 17), (7, 18), (8, 19), (9, 20)]
    invalid_transitions += [(12, 1), (13, 2), (14, 3), (15, 4), (17, 6), (18, 7), (19, 8), (20, 9)]
    invalid_transitions += [(56, 67), (57, 68), (58, 69), (59, 70), (61, 72), (62, 73), (63, 74), (64, 75)]
    invalid_transitions += [(67, 56), (68, 57), (69, 58), (70, 59), (72, 61), (73, 62), (74, 63), (75, 64)]
    
    # initialize world            
    world = make_gridworld(7, 11, terminals=[43, 33], rewards=np.array([[43, 1]]), goals=[43], starting_states=[71], invalid_transitions=invalid_transitions)    
    
    # a dictionary that contains all employed modules
    modules = {}
    modules['rl_interface'] = InterfaceGridworld(modules, world, False, None)
    
    # initialize RL agent
    rl_agent = SFMAAgent(modules['rl_interface'], 0.3, 5, 0.9, 0.9, gamma_SR=0.1)
    # common replay settings
    rl_agent.M.decay_inhibition = 0.9
    rl_agent.M.beta = 5
    rl_agent.M.mode = 'default'
    rl_agent.M.C_normalize = False
    rl_agent.M.D_normalize = False
    rl_agent.M.R_normalize = True
    
    # initialize experience
    for state in range(77):
        for action in range(4):
            rl_agent.M.states[state, action] = np.argmax(world['sas'][state, action])
    
    
    # start simulations           
    strengths = {}
    for condition in conditions:
        # reset relevant memory structures
        rl_agent.M.C.fill(0)
        rl_agent.M.I.fill(0)
        print('\t\tCondition: ' + condition)
        strengths[condition] = {}
        # start behavior
        rl_agent.M.C[run_patterns[conditions[condition][0]]] += 1.
        for t, trial in enumerate(conditions[condition][1:]):
            rl_agent.M.C[run_patterns[trial]] += 1.
            # determine the current state (i.e. follow-up state of last transition/experience)
            current_state = (run_patterns[trial][-1] % 77) + 11
            # store experience strength after first block
            if t == int(len(conditions[condition])/2):
                strengths[condition]['mid'] = np.copy(rl_agent.M.C)
        strengths[condition]['end'] = np.copy(rl_agent.M.C)
    
    return strengths


if __name__ == '__main__':
    # define behavior
    run_patterns = define_run_patterns()
    conditions = define_conditions(10)
    for condition in list(conditions.keys()):
        if not condition in ['RL', 'AA', 'RA', 'AL']:
            conditions.pop(condition)
    
    # make sure that the directory for storing the simulation results exists
    os.makedirs('data/', exist_ok=True)
    
    # run simulations
    print('Running simulations.')
    results = single_run(run_patterns, conditions)
    pickle.dump(results, open('data/strengths.pkl', 'wb'))
    