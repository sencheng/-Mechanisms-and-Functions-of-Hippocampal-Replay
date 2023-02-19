# basic imports
import os
import numpy as np
import pickle
import pyqtgraph as qg
# framework imports
from cobel.agents.sfma import SFMAAgent
from cobel.interfaces.gridworld import InterfaceGridworld
from cobel.analysis.rl_monitoring.rl_performance_monitors import RewardMonitor, EscapeLatencyMonitor
from cobel.misc.gridworld_tools import make_gridworld
# change directory
cwd = os.getcwd()

# shall the system provide visual output while performing the experiments?
# NOTE: do NOT use visualOutput=True in parallel experiments, visualOutput=True should only be used in explicit calls to 'singleRun'! 
visual_output = False


def define_run_patterns() -> dict:
    '''
    This function predefines run patterns for the agent (i.e. the chosen transition).
    The run patterns are defined so that they can used to reproduce the experimental
    conditions as described by Gupta et al. (2010).
    
    Parameters
    ----------
    None
    
    Returns
    ----------
    run_patterns :                      A dictionary containing the predefined run patterns.
    '''
    run_patterns = {}
    # start to left (used to reproduce alternating laps and left laps conditions)
    run_patterns['start_to_left'] = np.array([148, 137, 126, 115, 104, 93, 5, 4, 3, 2, 1, 231, 242, 253])
    # start to right (used to reproduce alternating laps and right laps condition)
    run_patterns['start_to_right'] = np.array([148, 137, 126, 115, 104, 93, 159, 160, 161, 162, 163, 241, 252, 263])
    # left arm to left arm (used to reproduce left laps condition)
    run_patterns['left_to_left'] = np.array([264, 275, 286, 220, 221, 222, 223, 224, 148, 137, 126, 115, 104, 93, 5, 4, 3, 2, 1, 231, 242, 253])
    # left arm to right arm (used to reproduce alternating laps condition)
    run_patterns['left_to_right'] = np.array([264, 275, 286, 220, 221, 222, 223, 224, 148, 137, 126, 115, 104, 93, 159, 160, 161, 162, 163, 241, 252, 263])
    # right arm to right arm (used to reproduce right laps condition)
    run_patterns['right_to_right'] = np.array([274, 285, 296, 76, 75, 74, 73, 72, 148, 137, 126, 115, 104, 93, 159, 160, 161, 162, 163, 241, 252, 263])
    # right arm to left arm (used to reproduce alternating laps condition)
    run_patterns['right_to_left'] = np.array([274, 285, 296, 76, 75, 74, 73, 72, 148, 137, 126, 115, 104, 93, 5, 4, 3, 2, 1, 231, 242, 253])
    
    return run_patterns

def single_run(run_patterns: dict, gamma_DR: float, mode: str, recency: bool, random_replay=False) -> (np.ndarray, np.ndarray):
    '''
    This function simulates a virutal version of the experimental paradigm by Gupta et al. (2010).
    An agent is trained to first run laps on one side of the maze, and then trained to run laps on the other side.
    
    Parameters
    ----------
    run_patterns :                      The set of predefined run patterns. Used to derive the action mask.
    gamma_DR :                          The discount factor used for the Default Representation.
    mode :                              The replay mode that will be used.
    recency :                           If true, priority ratings also depend on the recency of experience.
    random_replay :                     If true, experiences are replayed randomly.
    
    Returns
    ----------
    reward :                            The reward trace.
    escape_latency :                    The escape latency trace.
    '''
    np.random.seed()
    # this is the main window for visual output
    # normally, there is no visual output, so there is no need for an output window
    main_window = None
    # if visual output is required, activate an output window
    if visual_output:
        main_window = qg.GraphicsWindow(title='Simulation: Shortcuts Learning')
        
    # define walls
    invalid_transitions = []
    invalid_transitions += [(11, 12), (22, 23), (33, 34), (44, 45), (55, 56)]
    invalid_transitions += [(12, 11), (23, 22), (34, 33), (45, 44), (56, 55)]
    invalid_transitions += [(15, 16), (26, 27), (37, 38), (48, 49), (59, 60)]
    invalid_transitions += [(16, 15), (27, 26), (38, 37), (49, 48), (60, 59)]
    invalid_transitions += [(16, 17), (27, 28), (38, 39), (49, 50), (60, 61)]
    invalid_transitions += [(17, 16), (28, 27), (39, 38), (50, 49), (61, 60)]
    invalid_transitions += [(20, 21), (31, 32), (42, 43), (53, 54), (64, 65)]
    invalid_transitions += [(21, 20), (32, 31), (43, 42), (54, 53), (65, 64)]
    invalid_transitions += [(1, 12), (2, 13), (3, 14), (4, 15), (6, 17), (7, 18), (8, 19), (9, 20)]
    invalid_transitions += [(12, 1), (13, 2), (14, 3), (15, 4), (17, 6), (18, 7), (19, 8), (20, 9)]
    invalid_transitions += [(56, 67), (57, 68), (58, 69), (59, 70), (61, 72), (62, 73), (63, 74), (64, 75)]
    invalid_transitions += [(67, 56), (68, 57), (69, 58), (70, 59), (72, 61), (73, 62), (74, 63), (75, 64)]
    
    # initialize world            
    world = make_gridworld(7, 11, terminals=[43, 33], rewards=np.array([[43, 1]]), goals=[43], starting_states=[71], invalid_transitions=invalid_transitions)    
    
    # a dictionary that contains all employed modules
    modules = {}
    modules['rl_interface'] = InterfaceGridworld(modules, world, visual_output, main_window)
    
    # initialize performance Monitor
    escape_latency_monitor = EscapeLatencyMonitor(200, 30, main_window, visual_output)
    reward_monitor = RewardMonitor(200, main_window, visual_output, [0, 10])
    
    # initialize RL agent
    rl_agent = SFMAAgent(modules['rl_interface'], 0.3, 5, 0.9, 0.9, gamma_SR=gamma_DR,
                         custom_callbacks={'on_trial_end': [escape_latency_monitor.update, reward_monitor.update]})
    # common replay settings
    rl_agent.M.decay_inhibition = 0.9
    rl_agent.M.C_normalize = False
    rl_agent.M.D_normalize = False
    rl_agent.M.R_normalize = True
    rl_agent.gamma = 0.99
    rl_agent.mask_actions = True
    rl_agent.M.mode = mode
    rl_agent.M.recency = recency
    rl_agent.random_replay = random_replay
    
    # initialize experience
    for state in range(77):
        for action in range(4):
            rl_agent.M.states[state, action] = np.argmax(world['sas'][state, action])
    
    # initialize action mask
    rl_agent.action_mask.fill(False)    
    for pattern in run_patterns:
        for exp in run_patterns[pattern]:
            state = exp % 77
            action = int(exp/77)
            rl_agent.action_mask[state, action] = True
    
    # eventually, allow the OAI class to access the robotic agent class
    modules['rl_interface'].rl_agent = rl_agent
    
    # train agent
    # first block
    modules['rl_interface'].world['rewards'].fill(0)
    modules['rl_interface'].world['rewards'][33] = 1.
    rl_agent.train(number_of_trials=100, max_number_of_steps=30, replay_batch_size=20)
    # second block
    modules['rl_interface'].world['rewards'].fill(0)
    modules['rl_interface'].world['rewards'][43] = 1.
    rl_agent.train(number_of_trials=100, max_number_of_steps=30, replay_batch_size=20)
    
    # and also stop visualization
    if visual_output:
        main_window.close()
        
    return reward_monitor.reward_trace, escape_latency_monitor.latency_trace


if __name__ == '__main__':
    # params
    betas = np.linspace(5, 15, 11)
    modes = ['default', 'reverse']
    use_recency = [True, False]
    number_of_runs = 100
    # define behavior
    run_patterns = define_run_patterns()
    
    # low gamma
    print('Running simulations.')
    data = {}
    for mode in modes:
        data[mode] = {}
        for recency in use_recency:
            print('\tMode: ' + mode + ', Recency: ' + ('Yes' if recency else 'No'))
            R, S = [], []
            for i in range(number_of_runs):
                if (i + 1) % 10 == 0:
                    print('\t\tRun: ' + str(i + 1))
                rewards, steps = single_run(run_patterns, 0.1, mode, recency)
                R += [rewards]
                S += [steps]
            data[mode][recency] = {'rewards': R, 'steps': S}
    pickle.dump(data, open(cwd + '/data/sfma_learning.pkl', 'wb'))
    
    # random replay
    print('Running simulations with random replay.')
    R, S = [], []
    for i in range(number_of_runs):
        if (i + 1) % 10 == 0:
            print('\t\tRun: ' + str(i + 1))
        rewards, steps = single_run(run_patterns, 0.8, 'default', False, True)
        R += [rewards]
        S += [steps]
    pickle.dump({'rewards': R, 'steps': S}, open(cwd + '/data/random_learning.pkl', 'wb'))
