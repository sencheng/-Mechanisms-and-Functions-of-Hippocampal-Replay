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
    
    replays = {'start': [], 'end': []}
    modules['rl_interface'].first_run = True
    
    # define step end callback
    def step_end_callback(logs: dict):
        if logs['rl_parent'].interface_OAI.current_state == 9 and  logs['rl_parent'].interface_OAI.first_run:
            logs['rl_parent'].interface_OAI.first_run = False
            logs['rl_parent'].interface_OAI.world['rewards'][9] = 0.
            replays['end'].append(logs['rl_parent'].M.replay(replay_steps, 10))
            replays['start'].append(logs['rl_parent'].M.replay(replay_steps, 9))
            for exp in replays['end'][-1]:
                logs['rl_parent'].update_Q(exp)
    # define trial end callback
    def trial_end_callback(logs: dict):
        logs['rl_parent'].interface_OAI.world['rewards'][9] = 1.
        logs['rl_parent'].interface_OAI.first_run = True
    
    # register custom callbacks
    custom_callbacks = {'on_step_end': [step_end_callback], 'on_trial_end': [trial_end_callback], 'on_replay_end': [replay_monitor.update]}
    
    # initialize RL agent
    rl_agent = SFMAAgent(modules['rl_interface'], 0.3, 5, 0.9, 0.9, gamma_SR=gamma_DR, custom_callbacks=custom_callbacks)
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
    
    #print(len(replay_monitor.replay_traces['trial_begin']), len(replay_monitor.replay_traces['trial_end']))
    #print(len(replays['start']), len(replays['end']))
    
    # compile replays
    replays_start, replays_end = [], []
    for i in range(trials):
        replays_start.append(replay_monitor.replay_traces['trial_begin'][i])
        if len(replays['start']) > i:
            replays_start.append(replays['start'][i])
        replays_end.append(replay_monitor.replay_traces['trial_end'][i])
        if len(replays['end']) > i:
            replays_end.append(replays['end'][i])
        
    return {'trial_begin': replays_start, 'trial_end': replays_end}


if __name__ == '__main__':
    # params
    modes = ['default', 'reverse']
    number_of_runs = 100
    
    # initialize world           
    invalid_transitions = [(9, 10)]
    double_track = make_gridworld(1, 20, terminals=[0], rewards=np.array([[0, 1]]), invalid_transitions=invalid_transitions, goals=[0])
    double_track['starting_states'] = np.array([19])
    
    # make sure that the directory for storing the simulation results exists
    os.makedirs('data/', exist_ok=True)
    
    # run simulations in linear track environment
    print('Running Linear Track Simulations.')
    for mode in modes:
        print('\tReplay mode: ' + mode)
        file_name =  'data/double_track_mode_%s.pkl' % mode
        replays_start, replays_end = [], []
        for run in range(number_of_runs):
            if (run + 1) % 10 == 0:
                print('\t\tRun: ' + str(run + 1))
            traces = single_run(0.1, mode, double_track, 1000, 100, 10)
            replays_start += [traces['trial_begin']]
            replays_end += [traces['trial_end']]
            gc.collect()
        pickle.dump([replays_start, replays_end], open(file_name, 'wb'))
