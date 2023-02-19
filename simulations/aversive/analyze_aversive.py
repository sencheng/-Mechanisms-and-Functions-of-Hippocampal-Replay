# basic imports
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt


def recover_position(state: int, width: int) -> np.ndarray:
    '''
    This function recovers the position coordinates of a state.
    
    Parameters
    ----------
    state :                             The state index.
    width :                             The width of the environment.
    
    Returns
    ----------
    position :                          A numpy array containing the position coordinates.
    '''
    x = int(state/width)
    y = state % width
    
    return np.array([x, y])

def compute_reactivation_map(replays: list, dimensions: tuple) -> np.ndarray:
    '''
    This function computes the reactivation map for a list of replays.
    
    Parameters
    ----------
    replays :                           A list of replays.
    dimensions :                        The gridworld's dimensions.
    
    Returns
    ----------
    reactivation_map :                  The reactivation map.
    '''
    reactivation_map = np.zeros(dimensions)
    for run in replays:
        for replay in run:
            for experience in replay:
                x, y = recover_position(experience['state'], dimensions[1]).astype(int)
                reactivation_map[x, y] += 1
            
    return reactivation_map/np.sum(reactivation_map)

def compute_action_map(Q: list, dimensions: tuple) -> dict:
    '''
    This function computes the action maps for a list of Q-function traces.
    
    Parameters
    ----------
    Q :                                 A list of Q-function traces.
    dimensions :                        The gridworld's dimensions.
    
    Returns
    ----------
    action_maps :                       A dictionary containing the action maps.
    '''
    action_maps = {'start': np.zeros(dimensions), 'end': np.zeros(dimensions)}
    for run in Q:
        for trial in run[:5]:
            action_maps['start'] += (np.argmax(trial, axis=1) == 0).astype(float) * (np.amax(trial, axis=1) != 0).astype(float)
        for trial in run[-5:]:
            action_maps['end'] += (np.argmax(trial, axis=1) == 0).astype(float) * (np.amax(trial, axis=1) != 0).astype(float)
    
    action_maps['start'] /= len(Q) * 5
    action_maps['end'] /= len(Q) * 5
        
    return action_maps

def plot_reactivation_map(reactivation_map: np.ndarray):
    '''
    This function computes the reactivation map for a list of replays.
    
    Parameters
    ----------
    reactivation_map :                  A dictionary containing reactivation maps.
    
    Returns
    ----------
    None
    '''
    plt.figure(1, figsize=(20, 1))
    plt.title('Reactivation Map', fontsize=20)
    # heatmap of reactivation probabilities
    plt.pcolor(np.flip(reactivation_map, axis=0), cmap='hot', vmin=0, vmax=0.12)
    # plot walls
    plt.plot(np.array([0, 20, 20, 0, 0]), np.array([0, 0, 1, 1, 0]), color='cyan', linewidth=6)
    plt.xticks(np.arange(reactivation_map.shape[1]) + 0.5, np.arange(reactivation_map.shape[1]))
    plt.yticks(np.arange(reactivation_map.shape[0]) + 0.5, np.arange(reactivation_map.shape[0]))
    plt.colorbar()
    plt.savefig('plots/aversive_reactivation.png', dpi=200, bbox_inches='tight', transparent=True)
    plt.savefig('plots/aversive_reactivation.svg', dpi=200, bbox_inches='tight', transparent=True)
    plt.close('all')
    
def plot_action_maps(action_maps: dict):
    '''
    This function computes the reactivation map for a list of replays.
    
    Parameters
    ----------
    action_maps :                       A dictionary containing action maps.
    
    Returns
    ----------
    None
    '''
    titles = {'start': 'Early', 'end': 'Late'}
    plt.figure(1, figsize=(20, 8/3))
    plt.suptitle('Preference to Avert Shock Zone', fontsize=25, position=(0.45, 1.2))
    plt.subplots_adjust(hspace=1)
    for i, t in enumerate(action_maps):
        plt.subplot(2, 1, i + 1)
        plt.title(titles[t], fontsize=20)
        # heatmap of reactivation probabilities
        plt.pcolor(np.flip(action_maps[t], axis=0), cmap='hot', vmin=0, vmax=1)
        # plot walls
        plt.plot(np.array([0, 20, 20, 0, 0]), np.array([0, 0, 1, 1, 0]), color='cyan', linewidth=6)
        plt.xticks(np.arange(action_maps[t].shape[1]) + 0.5, np.arange(action_maps[t].shape[1]))
        plt.yticks(np.arange(action_maps[t].shape[0]) + 0.5, np.arange(action_maps[t].shape[0]))
        plt.colorbar()
    plt.savefig('plots/aversive_Q.png', dpi=200, bbox_inches='tight', transparent=True)
    plt.savefig('plots/aversive_Q.svg', dpi=200, bbox_inches='tight', transparent=True)
    plt.close('all')


if __name__ == '__main__':
    # make sure that the directory for storing the simulation results exists
    os.makedirs('plots/', exist_ok=True)
    
    # prepare data
    data = pickle.load(open('data/aversive_linear_track.pkl', 'rb'))
    reactivation_map = compute_reactivation_map(data['replays'], (1, 20))
    action_maps = compute_action_map(data['Q'], (1, 20))
    # plot data
    plot_reactivation_map(reactivation_map)
    plot_action_maps(action_maps)
    