# basic imports
import os
import numpy as np
import pickle
import pyqtgraph as qg
# tensorflow imports
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()
from tensorflow.keras import backend as K
# framework imports
from cobel.interfaces.oai_gym_gridworlds import OAIGymInterface
from cobel.analysis.rl_monitoring.rl_performance_monitors import EscapeLatencyMonitor
from cobel.misc.gridworld_tools import makeOpenField, makeGridworld
# change directory
cwd = os.getcwd()
os.chdir('../..')
# custom imports
from agents.deep_sfma_agent import DeepSFMAAgent

# shall the system provide visual output while performing the experiments?
# NOTE: do NOT use visualOutput=True in parallel experiments, visualOutput=True should only be used in explicit calls to 'singleRun'! 
visual_output = False


def single_run(world, mode):
    '''
    This method performs a single experimental run, i.e. one experiment.
    It has to be called by either a parallelization mechanism (without visual output),
    or by a direct call (in this case, visual output can be used).
    
    | **Args**
    | world:                        The gridworld environment.
    | mode:                         The replay mode that will be used.
    '''
    np.random.seed()
    # this is the main window for visual output
    # normally, there is no visual output, so there is no need for an output window
    main_window = None
    # if visual output is required, activate an output window
    if visual_output:
        main_window = qg.GraphicsWindow(title='Simulation: Approximation')    
    
    # a dictionary that contains all employed modules
    modules = {}
    modules['rl_interface'] = OAIGymInterface(modules, world, visual_output, main_window)
    
    # amount of trials
    number_of_trials = 500
    # max number of steps
    max_steps = 30
    
    # initialize performance Monitor
    escape_latency_monitor = EscapeLatencyMonitor(number_of_trials, max_steps, main_window, visual_output)
    
    # initialize RL agent
    rl_agent = DeepSFMAAgent(interface_OAI=modules['rl_interface'], epsilon=0.3, beta=5, learning_rate=0.9,
                                   gamma=0.9, gamma_SR=0.1, custom_callbacks={'on_trial_end': [escape_latency_monitor.update]})
    rl_agent.mask_actions = True
    rl_agent.random_replay = False
    rl_agent.epochs = 1
    rl_agent.replay_per_trial = 1
    if mode == 'dynamic':
        #rl_agent.dynamic_mode = True
        mode = 'default'
    rl_agent.M.decay_inhibition = 0.9
    rl_agent.M.beta = 9
    rl_agent.M.C_normalize = False
    rl_agent.M.D_normalize = False
    rl_agent.M.R_normalize = True
    rl_agent.M.mode = mode
    
    # eventually, allow the OAI class to access the robotic agent class
    modules['rl_interface'].rl_agent = rl_agent
    
    # let the agent learn
    rl_agent.train(number_of_trials, max_steps, 10)
    
    # clear session for performance
    K.clear_session()
    
    # and also stop visualization
    if visual_output:
        main_window.close()
    
    return escape_latency_monitor.latency_trace

if __name__ == '__main__':
    # params
    modes = ['default', 'reverse', 'dynamic']
    number_of_runs = 100
    
    # prepare worlds
    invalid_transitions = []
    invalid_transitions += [(0, 7), (1, 8), (2, 9), (4, 11), (5, 12), (6, 13)]
    invalid_transitions += [(7, 0), (8, 1), (9, 2), (11, 4), (12, 5), (13, 6)]
    invalid_transitions += [(9, 10), (16, 17), (23, 24), (30, 31), (37, 38), (44, 45), (51, 52), (58, 59), (65, 66)]
    invalid_transitions += [(10, 9), (17, 16), (24, 23), (31, 30), (38, 37), (45, 44), (52, 51), (59, 58), (66, 65)]
    invalid_transitions += [(10, 11), (17, 18), (24, 25), (31, 32), (38, 39), (45, 46), (52, 53), (59, 60), (66, 67)]
    invalid_transitions += [(11, 10), (18, 17), (25, 24), (32, 31), (39, 38), (46, 45), (53, 52), (60, 59), (67, 66)]           
    t_maze = makeGridworld(10, 7, terminals=[6], rewards=np.array([[6, 1]]), goals=[6], startingStates=[66], invalidTransitions=invalid_transitions)
    box = makeGridworld(10, 7, terminals=[6], rewards=np.array([[6, 1]]), goals=[6], startingStates=[66], invalidTransitions=[])
    
    # run simulations in T-maze environment
    print('Running simulations in T-maze.')
    steps = {}
    for mode in modes:
        print('\tMode: ' + str(mode))
        steps[mode] = []
        for run in range(number_of_runs):
            if (run + 1) % 10 == 0:
                print('\t\tRun: ' + str(run + 1))
            steps[mode] += [single_run(t_maze, mode)]
        fileName = cwd + '/data/approximation_tmaze_mode_' + mode + '.pkl'
        pickle.dump(steps, open(fileName, 'wb'))
            
    # run simulations in box environment
    print('Running simulations in box.')
    steps = {}
    for mode in modes:
        print('\tMode: ' + str(mode))
        steps[mode] = []
        for run in range(number_of_runs):
            if (run + 1) % 10 == 0:
                print('\t\tRun: ' + str(run + 1))
            steps[mode] += [single_run(box, mode)]
        fileName = cwd + '/data/approximation_box_mode_' + mode + '.pkl'
        pickle.dump(steps, open(fileName, 'wb'))
