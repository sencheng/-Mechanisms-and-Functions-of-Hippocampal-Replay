# basic imports
import os
import gc
import pickle
import numpy as np
import PyQt5 as qt
import pyqtgraph as qg
# framework imports
from cobel.agents.dyna_q import PMAAgent
from cobel.interfaces.gridworld import InterfaceGridworld
from cobel.misc.gridworld_tools import make_empty_field


def recover_positions(replay: list, width: int, height: int) -> list:
    '''
    This function recovers the coordiantes of replayed positions.
    
    Parameters
    ----------
    replay :                            A list of replayed experience tuples.
    width :                             The width of the environment in number of states.
    height :                            The height of the environment in number of states.
    
    Returns
    ----------
    positions :                         A list containing the coordinates of the replayed positions.
    '''
    positions = []
    for experience in replay:
        state = experience['state']
        x = int(state/height)
        y = state - x * height
        positions += [[x, y]]
    
    return positions

def single_run(number_of_replays: int, min_gain_mode: str = 'original') -> list:
    '''
    This method performs a single experimental run, i.e. one experiment.
    It has to be called by either a parallelization mechanism (without visual output),
    or by a direct call (in this case, visual output can be used).
    
    Parameters
    ----------
    number_of_replays :                 The number of replays that will be generated.
    min_gain_mode :                     The mode used when computing the gain term.
    
    Returns
    ----------
    replays :                           A list containing the generated replays.
    '''
    np.random.seed() 
    # initialize world            
    world = make_empty_field(20, 20)
    
    # a dictionary that contains all employed modules
    modules = dict()
    modules['rl_interface'] = InterfaceGridworld(modules, world, False, None)
    
    # initialize RL agent
    rl_agent = PMAAgent(interface_OAI=modules['rl_interface'], epsilon=0.3, beta=5, learning_rate=0.9, gamma=0.9)
    
    # eventually, allow the OAI class to access the robotic agent class
    modules['rl_interface'].rl_agent = rl_agent
        
    # initialize experiences
    for state in range(world['sas'].shape[0]):
        for action in range(world['sas'].shape[1]):
            rl_agent.M.states[state][action] = np.argmax(world['sas'][state][action])
    rl_agent.M.compute_update_mask()
    rl_agent.M.equal_need = True
    rl_agent.M.allow_loops = True
    rl_agent.M.ignore_barriers = False
    rl_agent.M.min_gain_mode = min_gain_mode
    
    replays =  []
    for i in range(number_of_replays):
        print('Replay: ', str(i+1))
        replays += [recover_positions(rl_agent.M.replay(50, 210), 20, 20)]
    
    return replays


if __name__ == '__main__':
    # params
    number_of_replays = 32
    
    # ensure directory for storing the replays exists
    os.makedirs('data/', exist_ok=True)
    
    # run simulation
    replays = single_run(number_of_replays) 
    pickle.dump(replays, open('data/replays_original.pkl', 'wb'))
    replays = single_run(number_of_replays, 'adjusted') 
    pickle.dump(replays, open('data/replays_adjusted.pkl', 'wb'))
    gc.collect()
    