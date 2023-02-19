# basic imports
import os
import pickle
import numpy as np
# framework imports
from cobel.agents.sfma import SFMAAgent
from cobel.interfaces.gridworld import InterfaceGridworld
from cobel.misc.gridworld_tools import make_open_field


def single_run(world: dict, number_of_replays: int = 20, batch_size: int = 20, mode: str = 'default') -> (np.ndarray, list):
    '''
    This method performs a single experimental run, i.e. one experiment.
    It has to be called by either a parallelization mechanism (without visual output),
    or by a direct call (in this case, visual output can be used).
    
    Parameters
    ----------
    world :                             The gridworld environment that the agent will be trained in.
    number_of_replays :                 The number of replays that will be generated.
    batch_size :                        The length of the replay batch.
    mode :                              The replay mode that will be used.
    
    Returns
    ----------
    latency_trace :                     The escape latency trace.
    mode_trace :                        The replay mode trace (reverse is coded as 1, default as 0).
    '''
    np.random.seed()
    # a dictionary that contains all employed modules
    modules = {}
    modules['rl_interface'] = InterfaceGridworld(modules, world, False, None)
    
    # define callback function for mode trace and reward changes
    mode_trace = []
    def trial_end_callback(logs: dict):
        mode_trace.append(int(logs['rl_parent'].M.mode == 'reverse'))
        if logs['trial'] == 100:
            logs['rl_parent'].interface_OAI.world['rewards'] *= 1.1
        elif logs['trial'] == 200:
            logs['rl_parent'].interface_OAI.world['rewards'] /= 1.1
        
    # initialize RL agent
    rl_agent = SFMAAgent(interface_OAI=modules['rl_interface'], epsilon=0.1, beta=5, learning_rate=0.9, gamma=0.99, gamma_SR=0.1)
    # common settings
    rl_agent.M.beta = 9
    rl_agent.M.C_step = 1.
    rl_agent.M.reward_mod = True
    rl_agent.M.D_normalize = False
    rl_agent.M.D_normalize = False
    rl_agent.M.R_normalize = True
    rl_agent.M.mode = mode
    rl_agent.mask_actions = True
    rl_agent.replays_per_trial = 1
    if mode == 'sweeping':
        rl_agent.M.decay_inhibition = 0.99
    
    # initialize experience
    rl_agent.M.C.fill(1.)
    for state in range(world['states']):
        for action in range(4):
            rl_agent.M.states[state, action] = np.argmax(world['sas'][state, action])
    
    # eventually, allow the OAI class to access the robotic agent class
    modules['rl_interface'].rl_agent = rl_agent
    
    # generate replays
    replays = [rl_agent.M.replay(batch_size, 60, 2) for replay in range(number_of_replays)]
    # retrieve similarity
    similarity = np.tile(rl_agent.M.metric.D[60], 4)
    if mode == 'reverse':
        similarity = similarity[rl_agent.M.states.flatten(order='F')]
    elif mode == 'forward':
        similarity = np.tile(rl_agent.M.metric.D[61], 4)
    elif mode == 'sweeping':
        similarity = np.tile(rl_agent.M.metric.D[61], 4)[rl_agent.M.states.flatten(order='F')]
    
    return similarity, replays


if __name__ == '__main__':
    # params
    number_of_replays, replay_length = 20, 10
    modes = ['default', 'reverse', 'forward', 'sweeping']
    world = make_open_field(11, 11, goal_state=0, reward=1)
    
    # make sure that the directory for storing the simulation results exists
    os.makedirs('data/', exist_ok=True)
    
    # run simulations
    results = {}
    print('Running simulations.')
    for mode in modes:
        print('\tMode: %s' % mode)
        DR, replays = single_run(world, number_of_replays, replay_length, mode)
        results[mode] = {'DR': DR, 'replay': replays}
    pickle.dump(results, open('data/results.pkl', 'wb'))
    