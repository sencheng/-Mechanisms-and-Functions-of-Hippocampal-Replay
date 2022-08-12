# basic imports
import os
import numpy as np
import pickle
import pyqtgraph as qg
# framework imports
from cobel.interfaces.oai_gym_gridworlds import OAIGymInterface
from cobel.misc.gridworld_tools import makeGridworld
# change directory
cwd = os.getcwd()
os.chdir('../..')
# custom imports
from agents.sfma_agent import SFMAAgent

# shall the system provide visual output while performing the experiments?
# NOTE: do NOT use visualOutput=True in parallel experiments, visualOutput=True should only be used in explicit calls to 'singleRun'! 
visual_output = False


def define_run_patterns():
    '''
    This function predefines run patterns for the agent (i.e. the chosen transition).
    The run patterns are defined so that they can used to reproduce the experimental
    conditions as described by Gupta et al. (2010).
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

def define_conditions(block_length=10):
    '''
    This function defines the different experimental conditions as described by Gupta et al. (2010).
    
    | **Args**
    | block_length:                 The number of trials a condition block lasts.
    '''
    # ensure valid block length
    if block_length == 1:
        block_length = 2
    elif block_length % 2 != 0:
        block_length += 1
    # define conditions
    conditions = {}
    # static conditions
    conditions['LL'] = ['start_to_left'] + ['left_to_left' for i in range(int(block_length * 2))]
    conditions['RR'] = ['start_to_right'] + ['right_to_right' for i in range(int(block_length * 2))]
    conditions['AA'] = ['start_to_right']
    for i in range(block_length):
        conditions['AA'] += ['right_to_left', 'left_to_right']
    # changing conditions
    conditions['LR'] = ['start_to_left'] + ['left_to_left' for i in range(block_length)] + ['left_to_right'] + ['right_to_right' for i in range(block_length - 1)]
    conditions['RL'] = ['start_to_right'] + ['right_to_right' for i in range(block_length)] + ['right_to_left'] + ['left_to_left' for i in range(block_length - 1)]
    conditions['LA'] = ['start_to_left'] + ['left_to_left' for i in range(block_length)]
    conditions['RA'] = ['start_to_right'] + ['right_to_right' for i in range(block_length)]
    conditions['AL'] = ['start_to_right']
    conditions['AR'] = ['start_to_right']
    for i in range(int(block_length/2)):
        conditions['LA'] += ['left_to_right', 'right_to_left']
        conditions['RA'] += ['right_to_left', 'left_to_right']
        conditions['AL'] += ['right_to_left', 'left_to_right']
        conditions['AR'] += ['right_to_left', 'left_to_right']
    conditions['AL'] += ['right_to_left'] + ['left_to_left' for i in range(block_length - 1)]
    conditions['AR'] += ['right_to_right' for i in range(block_length)]
    
    return conditions

def single_run(run_patterns, conditions, betas, gamma_DR, modes, use_recency, replays_per_trial=1, prefix=''):
    '''
    This function simulates a virutal version of the experimental paradigm by Gupta et al. (2010).
    Simulations are repeated for different model parameters (i.e. temperature, replay mode and effects of recency).
    The discount factor is kept fixed.
    
    | **Args**
    | run_patterns:                 The set of predefined run patterns.
    | conditions:                   The experimental conditions represented as lists of run patterns.
    | betas:                        A list of temperature parameters which controls the effect of priority rating during replay.
    | gamma_DR:                     The discount factor used for the Default Representation.
    | modes:                        A list of replay modes to be used.
    | use_recency:                  A list of recency flags. If the flag is true, the recency of experience will affect the priority rating.
    | replays_per_trial:            The number of replays that should generated after each replay.
    | prefix:                       Optional prefix which will be prepended to the file name.
    '''
    np.random.seed()
    # this is the main window for visual output
    # normally, there is no visual output, so there is no need for an output window
    main_window = None
    # if visual output is required, activate an output window
    if visual_output:
        main_window = qg.GraphicsWindow(title='Simulation: Shortcuts Static')
        
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
    world = makeGridworld(7, 11, terminals=[43, 33], rewards=np.array([[43, 1]]), goals=[43], startingStates=[71], invalidTransitions=invalid_transitions)    
    
    # a dictionary that contains all employed modules
    modules = {}
    modules['rl_interface'] = OAIGymInterface(modules, world, visual_output, main_window)
    
    # initialize RL agent
    rl_agent = SFMAAgent(modules['rl_interface'], 0.3, 5, 0.9, 0.9, gamma_SR=gamma_DR)
    # common replay settings
    rl_agent.M.decay_inhibition = 0.9
    rl_agent.M.C_normalize = False
    rl_agent.M.D_normalize = False
    rl_agent.M.R_normalize = True
    
    # initialize experience
    for state in range(77):
        for action in range(4):
            rl_agent.M.states[state, action] = np.argmax(world['sas'][state, action])

    # start simulations
    for mode in modes:
        for recency in use_recency:
            for beta in betas:
                print('\tReplay mode: ' + mode + ', Use recency: ' + ('Yes' if recency else 'No') + ', Beta: ' + str(beta))
                data = {}
                for condition in conditions:
                    # reset relevant memory structures
                    rl_agent.M.C.fill(0)
                    rl_agent.M.I.fill(0)
                    rl_agent.M.T.fill(0)
                    # specific replay settings
                    rl_agent.M.beta = beta
                    rl_agent.M.recency = recency
                    rl_agent.M.mode = mode
                    print('\t\tCondition: ' + condition)
                    data[condition] = []
                    # start behavior
                    rl_agent.M.C[run_patterns[conditions[condition][0]]] += 1.
                    rl_agent.M.T[run_patterns[conditions[condition][0]]] = np.flip(np.array([rl_agent.M.decay_recency ** i for i in range(len(run_patterns[conditions[condition][0]]))]))
                    for t, trial in enumerate(conditions[condition][1:]):
                        rl_agent.M.C[run_patterns[trial]] += 1.
                        rl_agent.M.T *= rl_agent.M.decay_recency ** len(run_patterns[trial])
                        rl_agent.M.T[run_patterns[trial]] = np.flip(np.array([rl_agent.M.decay_recency ** i for i in range(len(run_patterns[trial]))]))
                        # determine the current state (i.e. follow-up state of last transition/experience)
                        current_state = (run_patterns[trial][-1] % 77) + 11
                        # generate replays (starting at the end of the first session block)
                        if t >= int(len(conditions[condition])/2) - 1:
                            data[condition] += [[]]
                            for i in range(replays_per_trial):
                                replay = rl_agent.M.replay(18, current_state) # perform replay
                                #states = [exp['state'] for exp in replay]
                                data[condition][-1] += [replay] # store replay
                # save replays
                file_name = cwd + '/data/static/' + prefix + 'mode_' + mode + '_recency_' + ('Y' if recency else 'N') + '_beta_' + str(beta) + '.pkl'
                pickle.dump(data, open(file_name, 'wb'))
    
    # and also stop visualization
    if visual_output:
        main_window.close()


if __name__ == '__main__':
    # params
    betas = np.linspace(5, 15, 11)
    modes = ['default', 'reverse']
    use_recency = [True, False]
    replays_per_trial = 200
    # define behavior
    run_patterns = define_run_patterns()
    conditions = define_conditions(10)
    for condition in list(conditions.keys()):
        if not condition in ['RL', 'AA', 'RA', 'AL']:
            conditions.pop(condition)
    
    # run simulations
    print('Running simulations.')
    single_run(run_patterns, conditions, betas, 0.1, modes, use_recency, replays_per_trial)
