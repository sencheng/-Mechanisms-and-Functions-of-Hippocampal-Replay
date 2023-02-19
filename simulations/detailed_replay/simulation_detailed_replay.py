# basic imports
import os
import pickle
import numpy as np
# framework imports
from cobel.agents.sfma import SFMAAgent
from cobel.interfaces.gridworld import InterfaceGridworld
from cobel.misc.gridworld_tools import make_open_field
# local imports
from sfma_memory import SFMAMemory


def single_run(world: dict, batch_size: int = 20) -> (list, dict):
    '''
    This method performs a single experimental run, i.e. one experiment.
    It has to be called by either a parallelization mechanism (without visual output),
    or by a direct call (in this case, visual output can be used).
    
    Parameters
    ----------
    world :                             The gridworld environment that the agent will be trained in.
    batch_size :                        The length of the replay batch.
    
    Returns
    ----------
    replay :                            A list containing replayed experiences.
    var_trace :                         A dictionary containing traces of the prioritization variables.
    '''
    np.random.seed()
    # a dictionary that contains all employed modules
    modules = {}
    modules['rl_interface'] = InterfaceGridworld(modules, world, False, None)
        
    # initialize RL agent
    rl_agent = SFMAAgent(interface_OAI=modules['rl_interface'], epsilon=0.1, beta=5, learning_rate=0.9, gamma=0.99, gamma_SR=0.9)
    rl_agent.M = SFMAMemory(modules['rl_interface'], 4, 0.9)
    # common settings
    rl_agent.M.beta = 9
    rl_agent.M.C_step = 1.
    rl_agent.M.reward_mod = True
    rl_agent.M.D_normalize = False
    rl_agent.M.D_normalize = False
    rl_agent.M.R_normalize = True
    rl_agent.M.mode = 'default'
    rl_agent.mask_actions = True
    rl_agent.replays_per_trial = 1
    
    # initialize experience
    rl_agent.M.C.fill(1.)
    for state in range(world['states']):
        for action in range(4):
            rl_agent.M.states[state, action] = np.argmax(world['sas'][state, action])
    
    # eventually, allow the OAI class to access the robotic agent class
    modules['rl_interface'].rl_agent = rl_agent
    
    # generate replay
    replay = rl_agent.M.replay(batch_size, 12)
    
    return replay, rl_agent.M.var_trace


if __name__ == '__main__':
    # params
    replay_length = 10
    world = make_open_field(5, 5, goal_state=0, reward=1)
    
    # make sure that the directory for storing the simulation results exists
    os.makedirs('data/', exist_ok=True)
    
    # run simulations
    replay, var_traces = single_run(world, replay_length)
    pickle.dump(replay, open('data/replay.pkl', 'wb'))
    pickle.dump(var_traces, open('data/var_trace.pkl', 'wb'))
    