# basic imports
import os
import pickle
import numpy as np
# framework imports
from cobel.agents.sfma import SFMAAgent
from cobel.interfaces.gridworld import InterfaceGridworld
# local imports
from sfma_memory import SFMAMemoryNonLocal


def single_run(trials: list, replays_per_trial: int = 1, weighting_local: float = 0.) -> dict:
    '''
    This function simulates non-local replay in an open field environment.
    
    Parameters
    ----------
    trials :                            A list containing current locations.
    replays_per_trial :                 The number of replays that should generated after each replay.
    weighting_local :                   Scales the influence of the current location when initializing replay.
    
    Returns
    ----------
    None
    '''
    np.random.seed()
    # initialize world            
    world = pickle.load(open('environments/open_field.pkl', 'rb'))   
    
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
    results = {'current_states': [], 'replay_states': []}
    # reset relevant memory structures
    rl_agent.M.C.fill(1)
    rl_agent.M.I.fill(0)
    rl_agent.M.T.fill(0)
    # specific replay settings
    rl_agent.M.beta = 9
    rl_agent.M.mode = 'default'
    # start behavior
    for current_state in trials:
        results['current_states'].append(current_state)
        results['replay_states'].append([rl_agent.M.replay(18, current_state)[0]['state'] for i in range(replays_per_trial)])
            
    return results


if __name__ == '__main__':
    # params
    trials = [0, 9, 22, 27, 45, 54, 72, 77, 90, 99]
    replays_per_trial = 200
    weightings_local = [0, 5, 10]
    
    # make sure that the directory for storing the simulation results exists
    os.makedirs('data/', exist_ok=True)
    
    # run simulations
    print('Running simulations.')
    for weighting_local in weightings_local:
        print('\tWeighting Local: %d' % weighting_local)
        results = single_run(trials, replays_per_trial, weighting_local)
        pickle.dump(results, open('data/results_open_field_%d.pkl' % weighting_local, 'wb'))
