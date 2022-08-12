# basic imports
import os
import gc
import numpy as np
import pickle
import pyqtgraph as qg
# framework imports
from cobel.interfaces.oai_gym_gridworlds import OAIGymInterface
from cobel.misc.gridworld_tools import makeOpenField, makeGridworld
# change directory
cwd = os.getcwd()
os.chdir('../..')
# custom imports
from agents.sfma_agent import SFMAAgent
from monitors.replay_monitors import ReplayMonitor

# shall the system provide visual output while performing the experiments?
# NOTE: do NOT use visualOutput=True in parallel experiments, visualOutput=True should only be used in explicit calls to 'singleRun'! 
visual_output = False


def single_run(gamma_DR, mode, recency, world, steps, trials, replay_steps):
    '''
    This function simulates a virutal version of the experimental paradigm by Gupta et al. (2010).
    Simulations are repeated for different model parameters (i.e. temperature, replay mode and effects of recency).
    The discount factor is kept fixed.
    
    | **Args**
    | gamma_DR:                     The discount factor used for computing the default representation.
    | mode:                         The replay mode.
    | betas:                        A list of temperature parameters which controls the effect of priority rating during replay.
    | recency:                      If true, priority ratings also depend on the recency of experience.
    | world:                        The gridworld environment.
    | steps:                        The maximum number of steps per trial.
    | trial:                        The number of trial the agent is trained for.
    | replay_steps:                 The length of replays.
    '''
    np.random.seed()
    # this is the main window for visual output
    # normally, there is no visual output, so there is no need for an output window
    main_window = None
    # if visual output is required, activate an output window
    if visual_output:
        main_window = qg.GraphicsWindow(title='Simulation: Sequences')
    
    # a dictionary that contains all employed modules
    modules = {}
    modules['rl_interface'] = OAIGymInterface(modules, world, visual_output, main_window)
    
    # amount of trials
    number_of_trials = trials
    # max number of steps per trial
    max_steps = steps
    
    # initialize performance Monitor
    replay_monitor = ReplayMonitor(main_window, visual_output)
    
    # initialize RL agent
    rl_agent = SFMAAgent(modules['rl_interface'], 0.3, 5, 0.9, 0.9, gamma_SR=gamma_DR, custom_callbacks={'on_replay_end': [replay_monitor.update]})
    rl_agent.start_replay = True
    rl_agent.mask_actions = True
    rl_agent.gamma = 0.99
    # common replay settings
    rl_agent.M.decay_inhibition = 0.9
    rl_agent.M.beta = 8.
    rl_agent.M.C_normalize = False
    rl_agent.M.D_normalize = False
    rl_agent.M.R_normalize = True
    rl_agent.M.mode = mode
    rl_agent.M.recency = recency
    
    # eventually, allow the OAI class to access the robotic agent class
    modules['rl_interface'].rl_agent = rl_agent
    
    # let the agent learn
    rl_agent.train(trials, steps, replay_steps)
    
    # and also stop visualization
    if visual_output:
        main_window.close()
        
    return replay_monitor.replay_traces


if __name__ == '__main__':
    # params
    modes = ['default', 'reverse']
    use_recency = [True, False]
    number_of_runs = 100
    
    # initialize worlds            
    linear_track = makeOpenField(1, 10, goalState=0, reward=1)
    linear_track['startingStates'] = np.array([9])
    open_field = makeOpenField(10, 10, goalState=0, reward=1)
    open_field['startingStates'] = np.array([55])
    invalidTransitions = [(1, 2), (11, 12), (21, 22), (31, 32), (41, 42), (51, 52), (61, 62), (71, 72)]
    invalidTransitions += [(2, 1), (12, 11), (22, 21), (32, 31), (42, 41), (52, 51), (62, 61), (72, 71)]
    invalidTransitions += [(14, 24), (15, 25), (16, 26), (17, 27)]
    invalidTransitions += [(24, 14), (25, 15), (26, 16), (27, 17)]
    invalidTransitions += [(23, 24), (33, 34), (43, 44), (53, 54), (63, 64), (73, 74), (83, 84), (93, 94)]
    invalidTransitions += [(24, 23), (34, 33), (44, 43), (54, 53), (64, 63), (74, 73), (84, 83), (94, 93)]
    invalidTransitions += [(36, 46), (37, 47), (38, 48), (39, 49)]
    invalidTransitions += [(46, 36), (47, 37), (48, 38), (49, 39)]
    invalidTransitions += [(76, 86), (77, 87), (86, 76), (87, 77)]
    invalidTransitions += [(45, 46), (55, 56), (65, 66), (75, 76)]
    invalidTransitions += [(46, 45), (56, 55), (66, 65), (76, 75)]
    labyrinth = makeGridworld(10, 10, terminals=[0], rewards=np.array([[0, 1]]), invalidTransitions=invalidTransitions, goals=[0])
    labyrinth['startingStates'] = np.array([4])
    
    # run simulations in linear track environment
    print('Running Linear Track Simulations.')
    for mode in modes:
        for recency in use_recency:
            print('\tReplay mode: ' + mode + ', Use recency: ' + ('Yes' if recency else 'No'))
            fileName = cwd + '/data/linear_track_mode_' + mode + '_recency_' + ('Y' if recency else 'N') + '.pkl'
            replays_start, replays_end = [], []
            for run in range(number_of_runs):
                if (run + 1) % 10 == 0:
                    print('\t\tRun: ' + str(run + 1))
                traces = single_run(0.1, mode, recency, linear_track, 100, 100, 10)
                replays_start += [traces['trial_begin']]
                replays_end += [traces['trial_end']]
                gc.collect()
            pickle.dump([replays_start, replays_end], open(fileName, 'wb'))
            
    # run simulations in open field environment
    print('Running Open Field Simulations.')
    for mode in modes:
        for recency in use_recency:
            print('\tReplay mode: ' + mode + ', Use recency: ' + ('Yes' if recency else 'No'))
            fileName = cwd + '/data/open_field_mode_' + mode + '_recency_' + ('Y' if recency else 'N') + '.pkl'
            replays_start, replays_end = [], []
            for run in range(number_of_runs):
                if (run + 1) % 10 == 0:
                    print('\t\tRun: ' + str(run + 1))
                traces = single_run(0.1, mode, recency, open_field, 100, 100, 10)
                replays_start += [traces['trial_begin']]
                replays_end += [traces['trial_end']]
                gc.collect()
            pickle.dump([replays_start, replays_end], open(fileName, 'wb'))
            
    # run simulations in labyrinth environment
    print('Running Labyrinth Simulations.')
    for mode in modes:
        for recency in use_recency:
            print('\tReplay mode: ' + mode + ', Use recency: ' + ('Yes' if recency else 'No'))
            fileName = cwd + '/data/labyrinth_mode_' + mode + '_recency_' + ('Y' if recency else 'N') + '.pkl'
            replays_start, replays_end = [], []
            for run in range(number_of_runs):
                if (run + 1) % 10 == 0:
                    print('\t\tRun: ' + str(run + 1))
                traces = single_run(0.1, mode, recency, labyrinth, 300, 100, 50)
                replays_start += [traces['trial_begin']]
                replays_end += [traces['trial_end']]
                gc.collect()
            pickle.dump([replays_start, replays_end], open(fileName, 'wb'))
