# basic imports
import os
import pickle
import numpy as np
import pyqtgraph as qg
# tensorflow imports
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()
from tensorflow.keras import backend as K
# framework imports
from cobel.interfaces.oai_gym_gridworlds import OAIGymInterface
from cobel.analysis.rl_monitoring.rl_performance_monitors import EscapeLatencyMonitor
from cobel.misc.gridworld_tools import makeGridworld
# store experiment directory
cwd = os.getcwd()
# change directory
os.chdir('../..')
# custom imports
from agents.deep_sfma_agent import DeepSFMAAgent

# shall the system provide visual output while performing the experiments?
# NOTE: do NOT use visualOutput=True in parallel experiments, visualOutput=True should only be used in explicit calls to 'singleRun'! 
visual_output = False


def single_run(replay_type, number_of_replays, beta):
    '''
    This method performs a single experimental run, i.e. one experiment.
    It has to be called by either a parallelization mechanism (without visual output),
    or by a direct call (in this case, visual output can be used).
    
    | **Args**
    | replay_type:                  The type of replay used for model updates.
    | number_of_replays:            The number of replay per trial.
    | beta:                         The inverse temperature used for generating SFMA replays.
    '''
    np.random.seed()
    # this is the main window for visual output
    # normally, there is no visual output, so there is no need for an output window
    main_window = None
    # if visual output is required, activate an output window
    if visual_output:
        main_window = qg.GraphicsWindow(title='Simulation: Learning with Deep SFMA')
    
    # define environmental barriers
    invalid_transitions = [(3, 4), (4, 3), (8, 9), (9, 8), (13, 14), (14, 13), (18, 19), (19, 18)]
    invalid_transitions = []
    
    # initialize world
    world = makeGridworld(5, 5, terminals=[4], rewards=np.array([[4, 10]]), goals=[4], invalidTransitions=invalid_transitions)
    world['startingStates'] = np.array([12])
    
    # a dictionary that contains all employed modules
    modules = {}
    modules['rl_interface'] = OAIGymInterface(modules, world, visual_output, main_window)
    
    # amount of trials
    number_of_trials = 200
    # maximum steos per trial
    max_steps = 50
    
    # initialize performance Monitor
    escape_latency_monitor = EscapeLatencyMonitor(number_of_trials, max_steps, main_window, visual_output)
    
    # initialize RL agent
    rl_agent = DeepSFMAAgent(interface_OAI=modules['rl_interface'], epsilon=0.1, beta=5, learning_rate=0.9,
                             gamma=0.9, gamma_SR=0.1, custom_callbacks={'on_trial_end': [escape_latency_monitor.update]})
    rl_agent.M.mode = 'reverse'
    rl_agent.M.reward_mod = True
    rl_agent.M.beta = beta
    rl_agent.replays_per_trial = number_of_replays
    rl_agent.updates_per_replay = 1
    rl_agent.policy = 'softmax'
    rl_agent.local_targets = (replay_type == 'local')
    
    # eventually, allow the OAI class to access the robotic agent class
    modules['rl_interface'].rl_agent = rl_agent
    
    # let the agent learn
    rl_agent.train(number_of_trials, max_steps, replay_batch_size=10)
    
    # clear session for performance
    K.clear_session()
    
    # and also stop visualization
    if visual_output:
        main_window.close()
        
    return escape_latency_monitor.latency_trace

if __name__ == '__main__':
    # params
    betas = np.linspace(1, 15, 15)
    numbers_of_replays = [1, 10]
    replay_types = ['local', 'step']
    number_of_runs = 100
    
    # run simulations
    for replay_type in replay_types:
        for number_of_replays in numbers_of_replays:
            for beta in betas:
                print('Type: ', replay_type, 'Replays: ', number_of_replays, 'beta: ', beta)
                latency = []
                for run in range(number_of_runs):
                    print('\tRun: ', run+1)
                    latency.append(single_run(replay_type, number_of_replays, beta))
                file_name = cwd + '/data/type_' + replay_type + '_replays_' + str(number_of_replays) + '_beta_' + str(beta) + '.pkl'
                pickle.dump(latency, open(file_name, 'wb'))
    