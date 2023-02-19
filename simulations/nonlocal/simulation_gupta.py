# basic imports
import os
import pickle
import numpy as np
# framework imports
from cobel.agents.sfma import SFMAAgent
from cobel.interfaces.gridworld import InterfaceGridworld
# local imports
from sfma_memory import SFMAMemoryNonLocal


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

def single_run(run_patterns: dict, conditions: dict, replays_per_trial: int = 1, weighting_local: float = 0.) -> dict:
    '''
    This function simulates non-local replay in a virutal version of the experimental paradigm by Gupta et al. (2010).
    Simulations are repeated for different model parameters. The discount factor is kept fixed.
    
    Parameters
    ----------
    run_patterns :                      The set of predefined run patterns.
    conditions :                        The experimental conditions represented as lists of run patterns.
    replays_per_trial :                 The number of replays that should generated after each replay.
    weighting_local :                   Scales the influence of the current location when initializing replay.
    
    Returns
    ----------
    None
    '''
    np.random.seed()
    # initialize world            
    world = pickle.load(open('environments/eight_maze.pkl', 'rb'))   
    
    # a dictionary that contains all employed modules
    modules = {}
    modules['rl_interface'] = InterfaceGridworld(modules, world, False, None)
    
    # initialize RL agent
    rl_agent = SFMAAgent(modules['rl_interface'], 0.3, 5, 0.9, 0.9, gamma_SR=0.1)
    rl_agent.M = SFMAMemoryNonLocal(modules['rl_interface'], 4, 0.1)
    # replay settings
    rl_agent.M.decay_inhibition = 0.9
    rl_agent.M.weighting_local = weighting_local
    rl_agent.M.C_normalize = False
    rl_agent.M.D_normalize = False
    rl_agent.M.R_normalize = True
    
    # initialize experience
    for state in range(77):
        for action in range(4):
            rl_agent.M.states[state, action] = np.argmax(world['sas'][state, action])
    
    # collect replays
    results = {}
    for condition in conditions:
        results[condition] = {'current_states': [], 'replay_states': []}
        # reset relevant memory structures
        rl_agent.M.C.fill(0)
        rl_agent.M.I.fill(0)
        rl_agent.M.T.fill(0)
        # specific replay settings
        rl_agent.M.beta = 9
        rl_agent.M.mode = 'default'
        # start behavior
        rl_agent.M.C[run_patterns[conditions[condition][0]]] += 1.
        for t, trial in enumerate(conditions[condition][1:]):
            rl_agent.M.C[run_patterns[trial]] += 1.
            # determine the current state (i.e. follow-up state of last transition/experience)
            current_state = (run_patterns[trial][-1] % 77) + 11
            rl_agent.M.C += np.tile(rl_agent.M.metric.D[current_state], 4)
            results[condition]['current_states'].append(current_state)
            results[condition]['replay_states'].append([rl_agent.M.replay(18, current_state)[0]['state'] for i in range(replays_per_trial)])
            
    return results


if __name__ == '__main__':
    # params
    replays_per_trial = 200
    weightings_local = [0, 5, 10]
    # define behavior
    run_patterns = define_run_patterns()
    conditions = define_conditions(10)
    for condition in list(conditions.keys()):
        if not condition in ['RL', 'AA']:
            conditions.pop(condition)
    
    # make sure that the directory for storing the simulation results exists
    os.makedirs('data/', exist_ok=True)
    
    # run simulations
    print('Running simulations.')
    for weighting_local in weightings_local:
        print('\tWeighting Local: %d' % weighting_local)
        results = single_run(run_patterns, conditions, replays_per_trial, weighting_local)
        pickle.dump(results, open('data/results_gupta_%d.pkl' % weighting_local, 'wb'))
