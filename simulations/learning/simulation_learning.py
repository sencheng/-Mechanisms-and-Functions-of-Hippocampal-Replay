# basic imports
import os
import pickle
import numpy as np
import pyqtgraph as qg
# framework imports
from cobel.agents.dyna_q import PMAAgent
from cobel.agents.sfma import SFMAAgent
from cobel.interfaces.gridworld import InterfaceGridworld
from cobel.analysis.rl_monitoring.rl_performance_monitors import EscapeLatencyMonitor
# store experiment directory
cwd = os.getcwd()

# shall the system provide visual output while performing the experiments?
# NOTE: do NOT use visualOutput=True in parallel experiments, visualOutput=True should only be used in explicit calls to 'singleRun'! 
visual_output = False


def single_run(world: dict, no_replay=False, random_replay=False, gamma_SR=0.9, mode='default',
               trials=20, trial_steps=20, batch_size=20, agent_type='sfma', mask_actions=True) -> np.ndarray:
    '''
    This method performs a single experimental run, i.e. one experiment.
    It has to be called by either a parallelization mechanism (without visual output),
    or by a direct call (in this case, visual output can be used).
    
    Parameters
    ----------
    world :                             The grid world environment that the agent will be trained in.
    no_replay :                         If true, the agent will not perform experience replay.
    random_replay :                     If true, the agent will perform experience replay by sampling unformly from memory.
    gamma_SR :                          The discount factor used to compute the Successor Representation or Default Representation.
    mode :                              The SFMA's replay mode that will be used.
    trials :                            The number of trials that the agent will be trained for.
    trial_steps :                       The maximum number of steps per trial.
    batch_size :                        The length of the replay batch.
    agent_type :                        The type of agent to be used (sfma or pma).
    mask_actions :                      If true, the agent will ignore invalid actions (i.e. those for which the agent does not move).
    
    Returns
    ----------
    escape_latency :                    The escape latency trace.
    '''
    np.random.seed()
    # this is the main window for visual output
    # normally, there is no visual output, so there is no need for an output window
    main_window = None
    # if visual output is required, activate an output window
    if visual_output:
        main_window = qg.GraphicsWindow(title='Simulation: Learning')
    
    # a dictionary that contains all employed modules
    modules = {}
    modules['rl_interface'] = InterfaceGridworld(modules, world, visual_output, main_window)
    
    # initialize performance Monitor
    escape_latency_monitor = EscapeLatencyMonitor(trials, trial_steps, main_window, visual_output)
    
    rl_agent = None
    if agent_type == 'sfma':
        # initialize RL agent
        rl_agent = SFMAAgent(interface_OAI=modules['rl_interface'], epsilon=0.1, beta=5, learning_rate=0.9,
                             gamma=0.99, gamma_SR=gamma_SR, custom_callbacks={'on_trial_end': [escape_latency_monitor.update]})
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
        rl_agent.mask_actions = mask_actions
        rl_agent.replays_per_trial = 1
        rl_agent.random_replay = random_replay
    elif agent_type == 'pma':
        # initialize RL agent
        rl_agent = PMAAgent(interface_OAI=modules['rl_interface'], epsilon=0.1, beta=5,
                            learning_rate=0.9, gamma=0.99, custom_callbacks={'on_trial_end': [escape_latency_monitor.update]})
        rl_agent.M.beta = 9
        rl_agent.M.legacy_gain = True
        rl_agent.mask_actions = mask_actions
        # initialize experience
        for state in range(world['states']):
            for action in range(4):
                rl_agent.M.states[state, action] = np.argmax(world['sas'][state, action])
        rl_agent.M.compute_update_mask()
        # adjust batch size to account for replays at the beginning of trials
        batch_size = int(batch_size/2)
    
    # eventually, allow the OAI class to access the robotic agent class
    modules['rl_interface'].rl_agent = rl_agent
    
    # train agent
    rl_agent.train(trials, trial_steps, batch_size, no_replay)
    
    # and also stop visualization
    if visual_output:
        main_window.close()
    
    return escape_latency_monitor.latency_trace


if __name__ == '__main__':
    # params
    gammas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    number_of_runs = 100
    training_params = {'linear_track': [20, 100, 10], 'open_field': [100, 100, 10], 'labyrinth': [100, 300, 50]}
    
    # run simulations
    for environment in ['linear_track', 'open_field', 'labyrinth']:
        # load environment
        world = pickle.load(open(cwd + '/environments/' + environment + '.pkl', 'rb'))
        # retrieve training params
        trials, trial_steps, batch_size = training_params[environment]
        # run simulations for current environment
        print('Running simulations with ' + environment.replace('_', ' ') + ' environment.')
        sfma_def, sfma_rev, sfma_dyn, random, online, pma = {gamma: [] for gamma in gammas}, {gamma: [] for gamma in gammas}, {gamma: [] for gamma in gammas}, [], [], []
        for run in range(number_of_runs):
            if (run + 1) % 10 == 0:
                print('\tRun: ' + str(run + 1))
            for gamma in gammas:
                sfma_def[gamma] += [single_run(world, no_replay=False, random_replay=False, gamma_SR=gamma, mode='default', trials=trials, trial_steps=trial_steps, batch_size=batch_size, agent_type='sfma', mask_actions=True)]
                sfma_rev[gamma] += [single_run(world, no_replay=False, random_replay=False, gamma_SR=gamma, mode='reverse', trials=trials, trial_steps=trial_steps, batch_size=batch_size, agent_type='sfma', mask_actions=True)]
                sfma_dyn[gamma] += [single_run(world, no_replay=False, random_replay=False, gamma_SR=gamma, mode='dynamic', trials=trials, trial_steps=trial_steps, batch_size=batch_size, agent_type='sfma', mask_actions=True)]
            random += [single_run(world, no_replay=False, random_replay=True, gamma_SR=gamma, mode='default', trials=trials, trial_steps=trial_steps, batch_size=batch_size, agent_type='sfma', mask_actions=True)]
            online += [single_run(world, no_replay=True, random_replay=False, gamma_SR=gamma, mode='default', trials=trials, trial_steps=trial_steps, batch_size=batch_size, agent_type='sfma', mask_actions=True)]
            pma += [single_run(world, no_replay=False, random_replay=False, gamma_SR=0.9, mode='default', trials=trials, trial_steps=trial_steps, batch_size=batch_size, agent_type='pma', mask_actions=True)]
        pickle.dump(sfma_def, open(cwd + '/data/' + environment + '_SFMA_default.pkl', 'wb'))
        pickle.dump(sfma_rev, open(cwd + '/data/' + environment + '_SFMA_reverse.pkl', 'wb'))
        pickle.dump(sfma_dyn, open(cwd + '/data/' + environment + '_SFMA_dynamic.pkl', 'wb'))
        pickle.dump(random, open(cwd + '/data/' + environment + '_random.pkl', 'wb'))
        pickle.dump(online, open(cwd + '/data/' + environment + '_online.pkl', 'wb'))
        pickle.dump(pma, open(cwd + '/data/' + environment + '_PMA.pkl', 'wb'))
