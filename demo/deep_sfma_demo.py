# basic imports
import os
import numpy as np
import pyqtgraph as qg
# framework imports
from cobel.interfaces.oai_gym_gridworlds import OAIGymInterface
from cobel.analysis.rl_monitoring.rl_performance_monitors import EscapeLatencyMonitor
from cobel.misc.gridworld_tools import makeGridworld
# change directory
os.chdir('..')
# custom imports
from agents.deep_sfma_agent import DeepSFMAAgent

# shall the system provide visual output while performing the experiments?
# NOTE: do NOT use visualOutput=True in parallel experiments, visualOutput=True should only be used in explicit calls to 'singleRun'! 
visual_output = True


def single_run():
    '''
    This method performs a single experimental run, i.e. one experiment.
    It has to be called by either a parallelization mechanism (without visual output),
    or by a direct call (in this case, visual output can be used).
    '''
    np.random.seed()
    # this is the main window for visual output
    # normally, there is no visual output, so there is no need for an output window
    main_window = None
    # if visual output is required, activate an output window
    if visual_output:
        main_window = qg.GraphicsWindow(title='Demo: Deep SFMA Agent')
    
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
    number_of_trials = 100
    # maximum steos per trial
    max_steps = 50
    
    # initialize performance Monitor
    escape_latency_monitor = EscapeLatencyMonitor(number_of_trials, max_steps, main_window, visual_output)
    
    # initialize RL agent
    rl_agent = DeepSFMAAgent(interface_OAI=modules['rl_interface'], epsilon=0.1, beta=5, learning_rate=0.9,
                        gamma=0.9, gamma_SR=0.1, custom_callbacks={'on_trial_end': [escape_latency_monitor.update]})
    rl_agent.M.mode = 'reverse'
    rl_agent.M.reward_mod = True
    #rl_agent.mask_actions = True
    rl_agent.replays_per_trial = 10
    rl_agent.updates_per_replay = 1
    rl_agent.policy = 'softmax'
    rl_agent.local_targets = True
    #rl_agent.randomize_subsequent_replays = True
    #rl_agent.random_replay = True
    
    # eventually, allow the OAI class to access the robotic agent class
    modules['rl_interface'].rl_agent = rl_agent
    
    # let the agent learn
    rl_agent.train(number_of_trials, max_steps, replay_batch_size=10)
    
    # and also stop visualization
    if visual_output:
        main_window.close()

if __name__ == '__main__':
    single_run()
