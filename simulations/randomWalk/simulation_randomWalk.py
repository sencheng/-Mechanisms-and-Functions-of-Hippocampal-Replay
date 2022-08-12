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
# custom imports
from agents.sfma_agent import SFMAAgent

# shall the system provide visual output while performing the experiments?
# NOTE: do NOT use visualOutput=True in parallel experiments, visualOutput=True should only be used in explicit calls to 'singleRun'! 
visual_output = False


def single_run(gamma_SR, occupancies, modes, decays, number_of_replays, batch_size):
    '''
    This method performs a single experimental run, i.e. one experiment.
    It has to be called by either a parallelization mechanism (without visual output),
    or by a direct call (in this case, visual output can be used).
    
    | **Args**
    | gamma_SR:                     The discount factor used to compute the Successor Representation or Default Representation.
    | occupancies:                  Environment occupancies in form of predefined experience strengths.
    | modes:                        The SMA's replay modes that will be used.
    | decays:                       A list containing different inhibition decay factors.
    | number_of_replays:            The number of replays that will be generated per parameter combination.
    | batch_size:                   The length of the replay batch.
    '''
    np.random.seed()
    # this is the main window for visual output
    # normally, there is no visual output, so there is no need for an output window
    main_window = None
    # if visual output is required, activate an output window
    if visual_output:
        main_window = qg.GraphicsWindow(title='Simulation: Random Walk')
    
    # initialize world            
    world = makeEmptyField(100, 100)
    
    # a dictionary that contains all employed modules
    modules = {}
    modules['rl_interface'] = OAIGymInterface(modules, world, visual_output, main_window)
    
    # initialize RL agent
    rl_agent = SFMAAgent(interface_OAI=modules['rl_interface'], epsilon=0.3, beta=5,
                         learning_rate=0.9, gamma=0.9, gamma_SR=gamma_SR)
    rl_agent.M.C.fill(0.) 
    rl_agent.M.C_normalize = False
    rl_agent.M.D_normalize = False
    rl_agent.M.R_normalize = True
    rl_agent.M.beta = 9
    
    # eventually, allow the OAI class to access the robotic agent class
    modules['rl_interface'].rl_agent = rl_agent
    
    # and also stop visualization
    if visual_output:
        main_window.close()
        
    # initialize experiences
    for state in range(world['sas'].shape[0]):
        for action in range(world['sas'].shape[1]):
            rl_agent.M.states[state][action] = np.argmax(world['sas'][state][action])
    
    # simulate for a given DR discount factor
    for occupancy in occupancies:
        rl_agent.M.C = occupancies[occupancy]
        for mode in modes:
            rl_agent.M.mode = mode
            for decay in decays:
                rl_agent.M.decay_inhibition = decay
                replays = []
                print('Gamma: ' + str(gamma) + ', Occupancy: ' + occupancy + ', Mode: ' + mode + ', Decay: ' + str(decay))
                for replay in range(number_of_replays):
                    if (replay + 1) % 10 == 0:
                        print('\tReplay: ' + str(replay + 1))
                    replays += [rl_agent.M.replay(batch_size, 5050)]
                file_name = cwd + '/data/BDA/occupancy_' + occupancy + '_mode_' + mode + '_gamma_' + str(gamma) + '_decay_' + str(decay) + '.pkl'
                pickle.dump(replays, open(file_name, 'wb'))


if __name__ == '__main__':
    # params
    decays = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
    gammas = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.01]
    modes = ['reverse', 'default']
    number_of_replays = 50
    batch_size = 500
    occupancies = pickle.load(open(cwd + '/data/occupancies.pkl', 'rb'))
    
    # start simulations
    for gamma in gammas:
        single_run(gamma_SR=gamma, occupancies=occupancies, modes=modes, decays=decays, number_of_replays=number_of_replays, batch_size=batch_size) 
        gc.collect()
