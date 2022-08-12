# basic imports
import os
import gc
import pickle
import numpy as np
import pyqtgraph as qg
# framework imports
from cobel.interfaces.oai_gym_gridworlds import OAIGymInterface
from cobel.misc.gridworld_tools import makeEmptyField
# store experiment directory
cwd = os.getcwd()
# change directory
os.chdir('../..')
# custom framework
from agents.sfma_agent import SFMAAgent

# shall the system provide visual output while performing the experiments?
# NOTE: do NOT use visualOutput=True in parallel experiments, visualOutput=True should only be used in explicit calls to 'singleRun'! 
visual_output = False


def recover_positions(replay, width, height):
    '''
    This function recovers the coordiantes of replayed positions.
    
    | **Args**
    | replay:                       A list of replayed experience tuples.
    | width:                        The width of the environment in number of states.
    | height:                       The height of the environment in number of states.
    '''
    positions = []
    for experience in replay:
        state = experience['state']
        x = int(state/height)
        y = state - x * height
        positions += [[x, y]]
    
    return positions

def single_run(gamma_SR, decay, number_of_replays, occupancies, modes, betas):
    '''
    This method performs a single experimental run, i.e. one experiment.
    It has to be called by either a parallelization mechanism (without visual output),
    or by a direct call (in this case, visual output can be used).
    
    | **Args**
    | gamma_SR:                     The discount factor used to compute the Successor Representation or Default Representation.
    | decays:                       A list containing different inhibition decay factors.
    | number_of_replays:            The number of replays that will be generated per parameter combination.
    | occupancies:                  Environment occupancies in form of predefined experience strengths.
    | modes:                        The replay modes to be used.
    | betas:                        The inverse temperature to be used.
    '''
    np.random.seed()
    # this is the main window for visual output
    # normally, there is no visual output, so there is no need for an output window
    main_window = None
    # if visual output is required, activate an output window
    if visual_output:
        main_window = qg.GraphicsWindow(title='Simulation: Random Walk Distribution')
    
    # initialize world            
    world = makeEmptyField(100, 100)
    
    # a dictionary that contains all employed modules
    modules = {}
    modules['rl_interface'] = OAIGymInterface(modules, world, visual_output, main_window)
    
    # initialize RL agent
    rl_agent = SFMAAgent(interface_OAI=modules['rl_interface'], epsilon=0.3, beta=5,
                         learning_rate=0.9, gamma=0.9, gamma_SR=gamma_SR)
    rl_agent.M.C.fill(1)
    rl_agent.M.C_normalize = False
    rl_agent.M.D_normalize = False
    rl_agent.M.R_normalize = True
    rl_agent.M.beta = 5
    rl_agent.M.decay_inhibition = decay
    
    # eventually, allow the OAI class to access the robotic agent class
    modules['rl_interface'].rl_agent = rl_agent
    
    # and also stop visualization
    if visual_output:
        main_window.close()
        
    # initialize experiences
    for state in range(world['sas'].shape[0]):
        for action in range(world['sas'].shape[1]):
            rl_agent.M.states[state][action] = np.argmax(world['sas'][state][action])
    
    for occupancy in occupancies:
        rl_agent.M.C = occupancies[occupancy]
        for mode in modes:
            print('Occupancy: ' + occupancy + ', Mode: ' + mode)
            rl_agent.M.mode = mode
            for beta in betas:
                rl_agent.M.beta = beta
                replays = []
                for i in range(number_of_replays):
                    if (i + 1) % 100 == 0:
                        print('\tReplay: ' + str(i + 1))
                    replays += [recover_positions(rl_agent.M.replay(5), 100, 100)]
                file_name = cwd + '/data/distribution_occupancy_' + occupancy + '_mode_' + mode + '_beta_' + str(beta) +'.pkl'
                pickle.dump(replays, open(file_name, 'wb'))


if __name__ == '__main__':
    # params
    gamma = 0.9
    decay = 0.9
    number_of_replays = 10000
    occupancies = pickle.load(open(cwd + '/data/occupancies.pkl', 'rb'))
    modes = ['default', 'reverse']
    betas = [5, 10, 15]
    
    # run simulation
    single_run(gamma, decay, number_of_replays, occupancies, modes, betas) 
    gc.collect()
