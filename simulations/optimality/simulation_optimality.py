# basic imports
import os
import pickle
import numpy as np
import pyqtgraph as qg
# framework imports
from cobel.agents.sfma import SFMAAgent
from cobel.interfaces.gridworld import InterfaceGridworld
from cobel.analysis.rl_monitoring.replay_monitors import OptimalityMonitor
# store experiment directory
cwd = os.getcwd()


# shall the system provide visual output while performing the experiments?
# NOTE: do NOT use visualOutput=True in parallel experiments, visualOutput=True should only be used in explicit calls to 'singleRun'! 
visual_output = False


def single_run(world: dict, mode='default', trials=20, trial_steps=20, batch_size=20) -> (np.ndarray, np.ndarray):
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
    optimality_trace :                  The optimality trace.
    gain_trace :                        The gain trace.
    '''
    np.random.seed()
    # this is the main window for visual output
    # normally, there is no visual output, so there is no need for an output window
    main_window = None
    # if visual output is required, activate an output window
    if visual_output:
        main_window = qg.GraphicsWindow(title='Simulation: Optimality')
    
    # a dictionary that contains all employed modules
    modules = {}
    modules['rl_interface'] = InterfaceGridworld(modules, world, visual_output, main_window)
    
    # initialize performance Monitor
    optimality_monitor = OptimalityMonitor(trials, main_window, visual_output)
    #optimality_monitor.policy = 'softmax'
    
    # initialize RL agent
    rl_agent = SFMAAgent(interface_OAI=modules['rl_interface'], epsilon=0.1, beta=5, learning_rate=0.9, gamma=0.99, gamma_SR=0.1,
                         custom_callbacks={'on_replay_begin': [optimality_monitor.update_local_Q], 'on_replay_end': [optimality_monitor.update]})
    # common settings
    rl_agent.M.beta = 9
    rl_agent.M.C_step = 1.
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
    
    # and also stop visualization
    if visual_output:
        main_window.close()
    
    return optimality_monitor.optimality_trace, optimality_monitor.gain_trace


if __name__ == '__main__':
    # params
    gammas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    modes = ['default', 'reverse', 'dynamic']
    number_of_runs = 100
    training_params = {'linear_track': [20, 100, 10], 'open_field': [100, 100, 10], 'labyrinth': [100, 300, 50]}
    
    # run simulations
    for environment in training_params:
        # load environment
        world = pickle.load(open(cwd + '/environments/' + environment + '.pkl', 'rb'))
        # retrieve training params
        trials, trial_steps, batch_size = training_params[environment]
        # run simulations for current environment
        print('Running simulations with ' + environment.replace('_', ' ') + ' environment.')
        for mode in modes:
            print('\tReplay mode: ' + mode)
            opt, gain = [], []
            for run in range(number_of_runs):
                if (run + 1) % 10 == 0:
                    print('\t\tRun: ' + str(run + 1))
                o, g = single_run(world, mode=mode, trials=trials, trial_steps=trial_steps, batch_size=batch_size)
                opt.append(o)
                gain.append(g)
            file_name = cwd + '/data/optimality_' + environment + '_SFMA_' + mode + '.pkl'
            pickle.dump(np.array(opt), open(file_name, 'wb'))
            file_name = cwd + '/data/gain_' + environment + '_SFMA_' + mode + '.pkl'
            pickle.dump(np.array(gain), open(file_name, 'wb'))
        