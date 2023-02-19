# basic imports
import os
import pickle
import numpy as np
# framework imports
from cobel.agents.sfma import SFMAAgent
from cobel.interfaces.gridworld import InterfaceGridworld


def single_run(world: dict, trials: int = 20, batch_size: int = 20, number_of_replays: float = 20) -> (list, np.ndarray):
    '''
    This method performs a single experimental run, i.e. one experiment.
    It has to be called by either a parallelization mechanism (without visual output),
    or by a direct call (in this case, visual output can be used).
    
    Parameters
    ----------
    world :                             The gridworld environment that the agent will be trained in.
    trials :                            The number of trials that will be simulated.
    batch_size :                        The length of the replay batch.
    number_of_replays :                 The number of replays that will be generated after each trial.
    
    Returns
    ----------
    replays :                           A list of generated replays.
    Q :                                 A numpy array containing the Q-function trace.
    '''
    np.random.seed()
    # a dictionary that contains all employed modules
    modules = {}
    modules['rl_interface'] = InterfaceGridworld(modules, world, False, None)
        
    # initialize RL agent
    rl_agent = SFMAAgent(interface_OAI=modules['rl_interface'], epsilon=0.1, beta=5,
                         learning_rate=0.9, gamma=0.99, gamma_SR=0.1, custom_callbacks={})
    # common settings
    rl_agent.M.beta = 9
    rl_agent.M.C_step = 1.
    rl_agent.M.reward_mod = True
    rl_agent.M.D_normalize = False
    rl_agent.M.D_normalize = False
    rl_agent.M.R_normalize = True
    #rl_agent.dynamic_mode = True
    rl_agent.M.mode = 'default'
    rl_agent.mask_actions = True
    rl_agent.replays_per_trial = 1
    
    # eventually, allow the OAI class to access the robotic agent class
    modules['rl_interface'].rl_agent = rl_agent
    
    # define left and right trajectories
    left_dark = np.arange(9) + 1
    left_light = np.arange(9) + 10
    right_dark = np.arange(10) + 41
    right_light = np.arange(9) + 51
    
    # initialize experiences
    for state in range(world['states']):
        for action in range(4):
            rl_agent.M.states[state, action] = np.argmax(world['sas'][state, action])
    # set terminals and reward
    rl_agent.M.terminals.fill(1)
    rl_agent.M.terminals[2, 0] = 0
    # we reuse the Q-function to encode discounted cumulative punishment
    rl_agent.M.rewards[2, 0] = 1.
    
    # simulate behavior and record replays
    replays, Q = [], []
    # simulate exploration of the linear track during Run 1, Run 2 and Pre (Wu et al., 2017)
    dark_fraction = 0.74
    rl_agent.M.C[left_dark] += trials * dark_fraction * 3
    rl_agent.M.C[left_light] += trials * (1 - dark_fraction) * 3
    rl_agent.M.C[right_dark] += trials * dark_fraction * 3
    rl_agent.M.C[right_light] += trials * (1 - dark_fraction) * 3
    # initial trial with shock
    rl_agent.M.C[left_dark] += 1.
    rl_agent.M.C[left_light] += 1.
    rl_agent.M.C += np.tile(rl_agent.M.metric.D[1], 4)
    # generate replays and update the agent's Q-function
    for i in range(number_of_replays):
        replays.append(rl_agent.M.replay(batch_size, 1))
        for experience in replays[-1]:
            rl_agent.update_Q(experience)
    Q.append(np.copy(rl_agent.Q))
    # for the remaining trials the agent stops before the dark half
    for trial in range(trials - 1):
        rl_agent.M.C[left_light] += 1.
        rl_agent.M.C[right_light] += 1.
        # generate replays and update the agent's Q-function
        for i in range(number_of_replays):
            replays.append(rl_agent.M.replay(batch_size, 10))
            for experience in replays[-1]:
                rl_agent.update_Q(experience)
        Q.append(np.copy(rl_agent.Q))
    
    return replays, np.array(Q)


if __name__ == '__main__':
    # params
    number_of_runs = 100
    number_of_replays = 20
    world = pickle.load(open('environments/linear_track.pkl', 'rb'))
    
    # make sure that the directory for storing the simulation results exists
    os.makedirs('data/', exist_ok=True)
    
    # run simulations
    results = {'replays': [], 'Q': []}
    print('Running simulations.')
    for run in range(number_of_runs):
        if (run + 1) % 10 == 0:
            print('\tRun: %d' % (run + 1))
        R, Q = single_run(world, 10, 10, number_of_replays)
        results['replays'].append(R)
        results['Q'].append(Q)
    pickle.dump(results, open('data/aversive_linear_track.pkl', 'wb'))
    