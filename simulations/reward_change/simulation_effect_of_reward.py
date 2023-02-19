# basic imports
import os
import pickle
import numpy as np
# framework imports
from cobel.agents.sfma import SFMAAgent
from cobel.interfaces.gridworld import InterfaceGridworld
from cobel.analysis.rl_monitoring.rl_performance_monitors import EscapeLatencyMonitor


def single_run(world: dict, trials: int = 20, trial_steps: int = 20, batch_size: int = 20) -> (np.ndarray, np.ndarray):
    '''
    This method performs a single experimental run, i.e. one experiment.
    It has to be called by either a parallelization mechanism (without visual output),
    or by a direct call (in this case, visual output can be used).
    
    Parameters
    ----------
    world :                             The gridworld environment that the agent will be trained in.
    trials :                            The number of trials that the agent will be trained for.
    trial_steps :                       The maximum number of steps per trial.
    batch_size :                        The length of the replay batch.
    
    Returns
    ----------
    latency_trace :                     The escape latency trace.
    mode_trace :                        The replay mode trace (reverse is coded as 1, default as 0).
    '''
    np.random.seed()
    # a dictionary that contains all employed modules
    modules = {}
    modules['rl_interface'] = InterfaceGridworld(modules, world, False, None)
    
    # initialize performance Monitor
    el_monitor = EscapeLatencyMonitor(trials, trial_steps, None, False)
    
    # define callback function for mode trace and reward changes
    mode_trace = []
    def trial_end_callback(logs: dict):
        mode_trace.append(int(logs['rl_parent'].M.mode == 'reverse'))
        if logs['trial'] == 100:
            logs['rl_parent'].interface_OAI.world['rewards'] *= 1.1
        elif logs['trial'] == 200:
            logs['rl_parent'].interface_OAI.world['rewards'] /= 1.1
        
    # initialize RL agent
    rl_agent = SFMAAgent(interface_OAI=modules['rl_interface'], epsilon=0.1, beta=5, learning_rate=0.9, gamma=0.99,
                         gamma_SR=0.1, custom_callbacks={'on_replay_end': [el_monitor.update, trial_end_callback]})
    # common settings
    rl_agent.M.beta = 9
    rl_agent.M.C_step = 1.
    rl_agent.M.reward_mod = True
    rl_agent.M.D_normalize = False
    rl_agent.M.D_normalize = False
    rl_agent.M.R_normalize = True
    rl_agent.dynamic_mode = True
    rl_agent.M.mode = 'default'
    rl_agent.mask_actions = True
    rl_agent.replays_per_trial = 1
    
    # eventually, allow the OAI class to access the robotic agent class
    modules['rl_interface'].rl_agent = rl_agent
    
    # train agent
    rl_agent.train(trials, trial_steps, batch_size)
    
    return el_monitor.latency_trace , np.array(mode_trace)


if __name__ == '__main__':
    # params
    number_of_runs = 100
    worlds = {world: pickle.load(open('environments/%s.pkl' % world, 'rb')) for world in ['open_field', 't_maze']}
    training_params = {'open_field': [300, 100, 10], 't_maze': [300, 100, 10]}
    
    # make sure that the directory for storing the simulation results exists
    os.makedirs('data/', exist_ok=True)
    
    # run simulations
    for world in worlds:
        print('Running simulation in \'%s\'.' % world)
        # retrieve training params
        trials, trial_steps, batch_size = training_params[world]
        results = {'escape_latency': [], 'mode': []}
        for run in range(number_of_runs):
            if (run + 1) % 10 == 0:
                print('\tRun: %d' % (run + 1))
            latency_trace, mode_trace = single_run(worlds[world], trials, trial_steps, batch_size)
            results['escape_latency'].append(latency_trace)
            results['mode'].append(mode_trace)
        pickle.dump(results, open('data/%s.pkl' % world, 'wb'))
    