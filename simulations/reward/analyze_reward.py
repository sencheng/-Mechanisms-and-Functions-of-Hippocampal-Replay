# basic imports
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt


def recover_position(state: int, width: int) -> np.ndarray:
    '''
    This function recover the position coordinates of a state.
    
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

def plot_reactivation_maps(reactivation_maps: dict, world: str, mode: str):
    '''
    This function computes the reactivation map for a list of replays.
    
    Parameters
    ----------
    reactivation_maps :                 A dictionary containing the reactivation maps for replays recorded at the begin and end of a trials, and offline.
    world :                             The gridworld environment for which replays were recorded.
    mode :                              The replay mode that was used when generating replays.
    
    Returns
    ----------
    None
    '''
    # define subplot titles
    titles = {'trial_end': 'Online (Trial End)', 'trial_begin': 'Online (Trial Begin)', 'offline': 'Offline'}
    plt.figure(1, figsize=(15, 4))
    for i, replay_type in enumerate(reactivation_maps):
        plt.subplot(1, 3, i + 1)
        plt.title(titles[replay_type])
        # heatmap of reactivation probabilities
        plt.pcolor(np.flip(reactivation_maps[replay_type], axis=0), cmap='hot', vmin=0)
        # plot walls
        if world == 'open_field':
            plt.plot(np.array([0, 10, 10, 0, 0]), np.array([0, 0, 10, 10, 0]), color='cyan', linewidth=6)
        elif world == 't_maze':
            plt.plot(np.array([0, 3, 3, 4, 4, 7]), np.array([6, 6, 1, 1, 6, 6]), color='cyan', linewidth=3)
            plt.plot(np.array([0, 0, 7, 7]), np.array([6.1, 7, 7, 6.1]), color='cyan', linewidth=6)
        plt.xticks(np.arange(reactivation_maps[replay_type].shape[1]) + 0.5, np.arange(reactivation_maps[replay_type].shape[1]))
        plt.yticks(np.arange(reactivation_maps[replay_type].shape[0]) + 0.5, np.arange(reactivation_maps[replay_type].shape[0]))
        plt.colorbar()
    plt.savefig('plots/%s_%s.png' % (world, mode), dpi=200, bbox_inches='tight', transparent=True)
    plt.savefig('plots/%s_%s.svg' % (world, mode), dpi=200, bbox_inches='tight', transparent=True)
    plt.close('all')


if __name__ == '__main__':
     # params
    dimensions = {'open_field': (10, 10), 't_maze': (7, 7)}
    modes = ['default', 'reverse', 'dynamic']
    
    # make sure that the directory for storing the simulation results exists
    os.makedirs('plots/', exist_ok=True)
    
    # analyze and plot replays
    for world in ['open_field', 't_maze']:
        for mode in modes:
            data = pickle.load(open('data/%s_%s.pkl' % (world, mode), 'rb'))
            reactivation_maps = {replay_type: compute_reactivation_map(data[replay_type], dimensions[world]) for replay_type in data}
            plot_reactivation_maps(reactivation_maps, world, mode)
            