# basic imports
import numpy as np
import pickle
import matplotlib.pyplot as plt


def compute_action_probabilities(q: np.ndarray, beta: float) -> np.ndarray:
    '''
    This function computes the action selection probabilities for a given set of Q values under softmax policy.
    
    Parameters
    ----------
    q :                                 The Q values.
    beta :                              The softmax function's inverse temperature parameter.
    
    Returns
    ----------
    p :                                 The action selection probabilities.
    '''
    q = q[np.array([0, 2])]
    if np.sum(q == 0) == q.shape[0]:
        q.fill(1)
    p = np.exp(q * beta)
    p /= np.sum(p)
    
    return p

def recover_states(replay: list) -> np.ndarray:
    '''
    This function extracts the states of replayed experiences.
    
    Parameters
    ----------
    replay :                            A sequence of replayed experiences.
    
    Returns
    ----------
    states :                            The recovered sequence of replayed states.
    '''
    states = []
    for experience in replay:
        states += [experience['state']]
        
    return np.array(states)

def reactivations(replays: list) -> np.ndarray:
    '''
    This function computes a reactivation map from a list of replays.
    
    Parameters
    ----------
    replays :                           A list of replays.
    
    Returns
    ----------
    M :                                 The computed reactivation map.
    '''
    M = np.zeros((10, 7))
    for replay in replays:
        states = recover_states(replay)
        for state in states:
            x = int(state/7)
            y = state - 7 * x
            M[x, y] += 1.
            
    return M

def plot(M: np.ndarray, file_name: str):
    '''
    This function plots the reactivation map.
    
    Parameters
    ----------
    M :                                 The reactivation map.
    file_name :                         This name of the file that the plot will be saved as.
    
    Returns
    ----------
    None
    '''
    M /= np.sum(M)
    # define environmental borders
    wall_1_x, wall_1_y = [0, 7], [3, 3]
    wall_2_x, wall_2_y = [0, 3], [2, 2]
    wall_3_x, wall_3_y = [4, 7], [2, 2]
    wall_4_x, wall_4_y = [0, 0], [2.1, 3]
    wall_5_x, wall_5_y = [7, 7], [2.1, 3]
    wall_6_x, wall_6_y = [3.1, 3.9], [0, 0]
    wall_7_x, wall_7_y = [3, 3], [0, 2]
    wall_8_x, wall_8_y = [4, 4], [0, 2]
    # plot
    plt.figure(1, figsize=(8, 3))
    plt.pcolor(np.flip(M[:3, :] * 100, axis=0), cmap='hot', vmin=0, vmax=np.amax(M) * 0.1 * 100)
    plt.plot(wall_1_x, wall_1_y, color='c', linewidth=8, zorder=100)
    plt.plot(wall_2_x, wall_2_y, color='c', linewidth=4, zorder=100)
    plt.plot(wall_3_x, wall_3_y, color='c', linewidth=4, zorder=100)
    plt.plot(wall_4_x, wall_4_y, color='c', linewidth=8, zorder=100)
    plt.plot(wall_5_x, wall_5_y, color='c', linewidth=8, zorder=100)
    plt.plot(wall_6_x, wall_6_y, color='c', linewidth=8, zorder=100)
    plt.plot(wall_7_x, wall_7_y, color='c', linewidth=4, zorder=100)
    plt.plot(wall_8_x, wall_8_y, color='c', linewidth=4, zorder=100)
    plt.xticks(np.array([0.5, 3.5, 6.5]), np.array([1, 4, 7]))
    plt.yticks(np.array([0.5, 2.5]), np.array([8, 10]))
    plt.xlabel('X Position', fontsize=15)
    plt.ylabel('Y Position', fontsize=15)
    cbar = plt.colorbar()
    cbar.set_label('Fraction [%]', rotation=270, fontsize=15, labelpad=20)
    plt.savefig(file_name + '.png', dpi=200, bbox_inches='tight', transparent=True)
    plt.savefig(file_name + '.svg', dpi=200, bbox_inches='tight', transparent=True)
    plt.close('all')
    
def plot_fractions(R_cued: np.ndarray, suffix=''):
    '''
    This function plots the fractions of cued arm replays as a heatmap.
    
    Parameters
    ----------
    R_cued :                            The fractions.
    suffix :                            This suffix with which the plot will be saved as.
    
    Returns
    ----------
    None
    '''
    plt.figure(1, figsize=(5, 4))
    plt.pcolor(R_cued, cmap='hot', vmin=0)
    plt.title('Cued Arm Preplays', fontsize=16)
    plt.xlabel('Strength of Attention', fontsize=12)
    plt.ylabel('Asymmetric Allocation of Attention', fontsize=12)
    plt.xticks(np.array([0.5, 5.5, 10.5]), np.array([0.0, 0.1, 0.2]))
    plt.yticks(np.array([0.5, 5.5, 10.5]), np.array([0.5, 0.75, 1.0]))
    cbar = plt.colorbar()
    cbar.set_label('Fraction [%]', rotation=270, fontsize=10)
    plt.savefig('plots/preplay' + suffix + '.png', dpi=200, bbox_inches='tight', transparent=True)
    plt.savefig('plots/preplay' + suffix + '.svg', dpi=200, bbox_inches='tight', transparent=True)
    plt.close('all')


if __name__ == "__main__":
    # params
    modes = ['default', 'reverse']
    gammas = [0.1, 0.8]
    target = 7.37
    
    for gamma in gammas:
        for mode in modes:
            print('Discount factor: ' + str(gamma) + ', Mode: ' + mode)
            # load sweep data
            sweep = pickle.load(open('data/analyzed_preplay_gamma_' + str(gamma) + '_mode_' + mode + '.pkl', 'rb'))
            plot_fractions(sweep['preplay'], suffix='_gamma_' + str(gamma) + '_mode_' + mode)
            # determine parameters which best reproduce the fraction of preplays reported by Olafsdottir et al. (2015)
            fractions = np.abs(sweep['preplay'] - target)
            best = np.argmin(fractions)
            best_fraction, best_r = np.unravel_index(best, (11, 11))
            best_fraction, best_r = sweep['fractions'][best_fraction], sweep['R'][best_r]
            # load replays
            file_name = 'data/sweep/gamma_' + str(gamma) + '_mode_' + mode + '_arms_' + str(best_r) + '_cued_' + str(best_fraction) + '.pkl'
            replays = pickle.load(open(file_name, 'rb'))
            M = reactivations(replays)
            plot(M, 'plots/preplay_arm_reactivations_gamma_' + str(gamma) + '_mode_' + mode)
