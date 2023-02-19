# basic imports
import os
import gc
import pickle
import numpy as np
# framework imports
from cobel.agents.sfma import SFMAAgent
from cobel.analysis.rl_monitoring.replay_monitors import ReplayMonitor
from cobel.interfaces.gridworld import InterfaceGridworld
from cobel.misc.gridworld_tools import make_open_field, make_gridworld


def single_run(gamma_DR: float, mode: str, world: dict, steps: int, trials: int, replay_steps: int) -> dict:
    '''
    This method performs a single experimental run, i.e. one experiment.
    It has to be called by either a parallelization mechanism (without visual output),
    or by a direct call (in this case, visual output can be used).
    
    Parameters
    ----------
    gamma_DR :                          The discount factor used for computing the default representation.
    mode :                              The replay mode.
    world :                             The gridworld environment.
    steps :                             The maximum number of steps per trial.
    trial :                             The number of trial the agent is trained for.
    replay_steps :                      The length of replays.
    
    Returns
    ----------
    replay_traces :                     The replay traces.
    '''
    # a dictionary that contains all employed modules
    modules = {}
    modules['rl_interface'] = InterfaceGridworld(modules, world, False, None)
    
    # initialize performance Monitor
    replay_monitor = ReplayMonitor(None, False)
    
    # initialize RL agent
    rl_agent = SFMAAgent(modules['rl_interface'], 0.3, 5, 0.9, 0.9, gamma_SR=gamma_DR, custom_callbacks={'on_replay_end': [replay_monitor.update]})
    rl_agent.start_replay = True
    rl_agent.mask_actions = True
    rl_agent.gamma = 0.99
    # common replay settings
    rl_agent.M.decay_inhibition = 0.9
    rl_agent.M.beta = 9.
    rl_agent.M.C_normalize = False
    rl_agent.M.D_normalize = False
    rl_agent.M.R_normalize = True
    rl_agent.M.mode = mode
    rl_agent.M.recency = False
    
    # eventually, allow the OAI class to access the robotic agent class
    modules['rl_interface'].rl_agent = rl_agent
    
    # let the agent learn
    rl_agent.train(trials, steps, replay_steps)
        
    return replay_monitor.replay_traces


if __name__ == '__main__':
    # params
    modes = ['default', 'reverse']
    number_of_runs = 100
    
    # initialize worlds            
    linear_track = make_open_field(1, 10, goal_state=0, reward=1)
    linear_track['starting_states'] = np.array([9])
    open_field = make_open_field(10, 10, goal_state=0, reward=1)
    open_field['starting_states'] = np.array([55])
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
    labyrinth = make_gridworld(10, 10, terminals=[0], rewards=np.array([[0, 1]]), invalid_transitions=invalidTransitions, goals=[0])
    labyrinth['starting_states'] = np.array([4])
    
    # make sure that the directory for storing the simulation results exists
    os.makedirs('data/', exist_ok=True)
    
    # run simulations in linear track environment
    print('Running Linear Track Simulations.')
    for mode in modes:
        print('\tReplay mode: ' + mode)
        file_name =  'data/linear_track_mode_' + mode + '.pkl'
        replays_start, replays_end = [], []
        for run in range(number_of_runs):
            if (run + 1) % 10 == 0:
                print('\t\tRun: ' + str(run + 1))
            traces = single_run(0.1, mode, linear_track, 100, 100, 10)
            replays_start += [traces['trial_begin']]
            replays_end += [traces['trial_end']]
            gc.collect()
        pickle.dump([replays_start, replays_end], open(file_name, 'wb'))
            
    # run simulations in open field environment
    print('Running Open Field Simulations.')
    for mode in modes:
        print('\tReplay mode: ' + mode)
        file_name = 'data/open_field_mode_' + mode + '.pkl'
        replays_start, replays_end = [], []
        for run in range(number_of_runs):
            if (run + 1) % 10 == 0:
                print('\t\tRun: ' + str(run + 1))
            traces = single_run(0.1, mode, open_field, 100, 100, 10)
            replays_start += [traces['trial_begin']]
            replays_end += [traces['trial_end']]
            gc.collect()
        pickle.dump([replays_start, replays_end], open(file_name, 'wb'))
            
    # run simulations in labyrinth environment
    print('Running Labyrinth Simulations.')
    for mode in modes:
        print('\tReplay mode: ' + mode)
        file_name = 'data/labyrinth_mode_' + mode + '.pkl'
        replays_start, replays_end = [], []
        for run in range(number_of_runs):
            if (run + 1) % 10 == 0:
                print('\t\tRun: ' + str(run + 1))
            traces = single_run(0.1, mode, labyrinth, 300, 100, 50)
            replays_start += [traces['trial_begin']]
            replays_end += [traces['trial_end']]
            gc.collect()
        pickle.dump([replays_start, replays_end], open(file_name, 'wb'))
