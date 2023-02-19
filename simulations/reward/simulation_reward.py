# basic imports
import os
import pickle
import numpy as np
# framework imports
from cobel.agents.sfma import SFMAAgent
from cobel.interfaces.gridworld import InterfaceGridworld
from cobel.analysis.rl_monitoring.replay_monitors import ReplayMonitor


def single_run(world: dict, mode: str = 'default', trials: int = 20, trial_steps: int = 20, batch_size: int = 20) -> dict:
    '''
    This method performs a single experimental run, i.e. one experiment.
    It has to be called by either a parallelization mechanism (without visual output),
    or by a direct call (in this case, visual output can be used).
    
    Parameters
    ----------
    world :                             The gridworld environment that the agent will be trained in.
    mode :                              The SFMA's replay mode that will be used.
    trials :                            The number of trials that the agent will be trained for.
    trial_steps :                       The maximum number of steps per trial.
    batch_size :                        The length of the replay batch.
    
    Returns
    ----------
    replay_traces :                     A dictionary containing the replays generated.
    '''
    np.random.seed()
    # a dictionary that contains all employed modules
    modules = {}
    modules['rl_interface'] = InterfaceGridworld(modules, world, False, None)
    
    # initialize performance Monitor
    replay_monitor = ReplayMonitor(None, False)
    
    # define callback function for offline and trial begin replays
    replay_traces = {'trial_begin': [], 'offline': []}
    def trial_end_callback(logs: dict):
        replay_traces['trial_begin'].append(logs['rl_parent'].M.replay(batch_size, world['starting_states'][0]))
        replay_traces['offline'].append(logs['rl_parent'].M.replay(batch_size))    
    
    # initialize RL agent
    rl_agent = SFMAAgent(interface_OAI=modules['rl_interface'], epsilon=0.1, beta=5, learning_rate=0.9, gamma=0.99,
                         gamma_SR=0.1, custom_callbacks={'on_replay_end': [replay_monitor.update, trial_end_callback]})
    # common settings
    rl_agent.M.beta = 9
    rl_agent.M.C_step = 1.
    rl_agent.M.reward_modulation = 10.
    rl_agent.M.reward_mod = True
    rl_agent.M.D_normalize = False
    rl_agent.M.D_normalize = False
    rl_agent.M.R_normalize = True
    # simulation specific settings
    if mode == 'dynamic':
        mode = 'default'
        rl_agent.dynamic_mode = True
    rl_agent.M.mode = mode
    rl_agent.mask_actions = True
    rl_agent.replays_per_trial = 1
    
    # eventually, allow the OAI class to access the robotic agent class
    modules['rl_interface'].rl_agent = rl_agent
    
    # train agent
    rl_agent.train(trials, trial_steps, batch_size)

    # combine recorded traces
    replay_traces['trial_end'] = replay_monitor.replay_traces['trial_end']
    
    return replay_traces


if __name__ == '__main__':
    # params
    number_of_runs = 100
    modes = ['default', 'reverse', 'dynamic']
    worlds = {world: pickle.load(open('environments/%s.pkl' % world, 'rb')) for world in ['open_field', 't_maze']}
    training_params = {'open_field': [100, 100, 10], 't_maze': [100, 100, 10]}
    
    # make sure that the directory for storing the simulation results exists
    os.makedirs('data/', exist_ok=True)
    
    # run simulations
    for world in worlds:
        # retrieve training params
        trials, trial_steps, batch_size = training_params[world]
        for mode in modes:
            print('Running simulation in \'%s\' with %s mode.' % (world, mode))
            R = {replay_type: [] for replay_type in ['trial_begin', 'trial_end', 'offline']}
            for run in range(number_of_runs):
                if (run + 1) % 10 == 0:
                    print('\tRun: %d' % (run + 1))
                replay_traces = single_run(worlds[world], mode, trials, trial_steps, batch_size)
                for trial_type in R:
                    R[trial_type].append(replay_traces[trial_type])
            pickle.dump(R, open('data/%s_%s.pkl' % (world, mode), 'wb'))
    