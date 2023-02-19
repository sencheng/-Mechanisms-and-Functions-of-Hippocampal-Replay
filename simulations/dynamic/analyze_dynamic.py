# basic imports
import numpy as np
import pickle
import matplotlib.pyplot as plt


def computeReactivationMaps(euclidean_similarity=False, suffix='') -> np.ndarray:
    '''
    This function computes the reactivation maps for replays recorded
    in the experiment's three environments.
    
    Parameters
    ----------
    euclidean_similarity :              If true, then reactivation maps for euclidean distance metric are loaded.
    suffix :                            The suffix of the file that will be loaded.
    
    Returns
    ----------
    reactivation_maps :                 The reactivation maps.
    '''
    # prepare reactivation maps
    M = np.zeros((3, 10, 10))
    steps = [8, 50, 15]
    steps = [10, 10, 10]
    # load replays
    fileName = 'data/replays_' + ('euclidean' if euclidean_similarity else 'DR') + suffix + '.pkl'
    replays = pickle.load(open(fileName, 'rb'))
    for i, env in enumerate(replays):
        # count reactivations
        for replay in replays[env]:
            for experience in replay[:steps[i]]:
                state = experience['state']
                x = int(state/10)
                y = state - x * 10
                M[i, x, y] += 1
        # compute fractions
        M[i, :, :] /= np.sum(M[i, :, :])
        
    return M

def plot(reactivation_maps: np.ndarray, euclidean_similarity=False, suffix=''):
    '''
    This functions plots the reactivation maps for the experiment's three environments.
    
    Parameters
    ----------
    reactivation_maps :                 The reactivation maps.
    euclidean_similarity :              If true, then reactivation maps for euclidean distance metric are loaded.
    suffix :                            The suffix of the file that will be saved.
    
    Returns
    ----------
    None
    '''
    # define environmental borders
    x1, y1 = [0, 0, 10, 10], [0, 10, 10, 0]
    x2, y2 = [0, 10, 10, 0], [10, 10, 0, 0]
    plt.figure(1, figsize=(16.5, 4))
    plt.suptitle('Reactivation Maps', fontsize=30, position=(0.5, 1.15))
    plt.subplots_adjust(wspace=0.3)
    # environment 1
    plt.subplot(1, 3, 1)
    plt.pcolor(np.flip(reactivation_maps[0], axis=0), cmap='hot', vmin=0, vmax=None)
    plt.plot(x1, y1, x2, y2, color='c', linewidth=10, zorder=100)
    plt.title('Environment 1', fontsize=25, position=(0.5, 1.15))
    plt.xlabel('X Position', fontsize=20)
    plt.ylabel('Y Position', fontsize=20)
    plt.xticks([0.5, 4.5, 9.5], [1, 5, 10])
    plt.yticks([0.5, 4.5, 9.5], [1, 5, 10])
    plt.xticks([])
    plt.yticks([])
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=15)
    # environment 2
    plt.subplot(1, 3, 2)
    wall_1_x, wall_1_y = [2, 2], [2, 10]
    wall_2_x, wall_2_y = [4, 4], [0, 8]
    wall_3_x, wall_3_y = [6, 6], [2, 6]
    wall_4_x, wall_4_y = [8, 8], [2, 4]
    wall_5_x, wall_5_y = [4, 8], [8, 8]
    wall_6_x, wall_6_y = [6, 10], [6, 6]
    wall_7_x, wall_7_y = [6, 8], [2, 2]
    plt.pcolor(np.flip(reactivation_maps[1], axis=0), cmap='hot', vmin=0, vmax=None)
    plt.plot(x1, y1, x2, y2, color='c', linewidth=10, zorder=100)
    plt.plot(wall_1_x, wall_1_y, color='c', linewidth=5, zorder=100)
    plt.plot(wall_2_x, wall_2_y, color='c', linewidth=5, zorder=100)
    plt.plot(wall_3_x, wall_3_y, color='c', linewidth=5, zorder=100)
    plt.plot(wall_4_x, wall_4_y, color='c', linewidth=5, zorder=100)
    plt.plot(wall_5_x, wall_5_y, color='c', linewidth=5, zorder=100)
    plt.plot(wall_6_x, wall_6_y, color='c', linewidth=5, zorder=100)
    plt.plot(wall_7_x, wall_7_y, color='c', linewidth=5, zorder=100)
    plt.title('Environment 2', fontsize=25, position=(0.5, 1.15))
    plt.xlabel('X Position', fontsize=20)
    plt.ylabel('Y Position', fontsize=20)
    plt.xticks([0.5, 4.5, 9.5], [1, 5, 10])
    plt.yticks([0.5, 4.5, 9.5], [1, 5, 10])
    plt.xticks([])
    plt.yticks([])
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=15)
    # environment 3
    plt.subplot(1, 3, 3)
    wall_1_x, wall_1_y = [5, 5], [0, 4]
    wall_2_x, wall_2_y = [5, 5], [6, 10]
    wall_3_x, wall_3_y = [3, 3], [2, 4]
    wall_4_x, wall_4_y = [3, 3], [6, 8]
    wall_5_x, wall_5_y = [7, 7], [2, 4]
    wall_6_x, wall_6_y = [7, 7], [6, 8]
    wall_7_x, wall_7_y = [3, 7], [4, 4]
    wall_8_x, wall_8_y = [3, 7], [6, 6]
    plt.pcolor(np.flip(reactivation_maps[2], axis=0), cmap='hot', vmin=0, vmax=None)
    plt.plot(x1, y1, x2, y2, color='c', linewidth=10, zorder=100)
    plt.plot(wall_1_x, wall_1_y, color='c', linewidth=5, zorder=100)
    plt.plot(wall_2_x, wall_2_y, color='c', linewidth=5, zorder=100)
    plt.plot(wall_3_x, wall_3_y, color='c', linewidth=5, zorder=100)
    plt.plot(wall_4_x, wall_4_y, color='c', linewidth=5, zorder=100)
    plt.plot(wall_5_x, wall_5_y, color='c', linewidth=5, zorder=100)
    plt.plot(wall_6_x, wall_6_y, color='c', linewidth=5, zorder=100)
    plt.plot(wall_7_x, wall_7_y, color='c', linewidth=5, zorder=100)
    plt.plot(wall_8_x, wall_8_y, color='c', linewidth=5, zorder=100)
    plt.title('Environment 3', fontsize=25, position=(0.5, 1.15))
    plt.xlabel('X Position', fontsize=20)
    plt.ylabel('Y Position', fontsize=20)
    plt.xticks([0.5, 4.5, 9.5], [1, 5, 10])
    plt.yticks([0.5, 4.5, 9.5], [1, 5, 10])
    plt.xticks([])
    plt.yticks([])
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=15)
    plt.savefig('plots/transparent_dynamic_reactivation_maps_' + ('euclidean' if euclidean_similarity else 'DR') + suffix + '.png', dpi=200, bbox_inches='tight', transparent=True)
    plt.savefig('plots/transparent_dynamic_reactivation_maps_' + ('euclidean' if euclidean_similarity else 'DR') + suffix + '.svg', dpi=200, bbox_inches='tight', transparent=True)
    plt.close('all')


if __name__ == "__main__":
    # analyze simulations using DR similarity
    M = computeReactivationMaps(suffix='_default')
    plot(M, suffix='_default')
    M = computeReactivationMaps(suffix='_reverse')
    plot(M, suffix='_reverse')
    # analyze simulations using Euclidean similarity
    M = computeReactivationMaps(True)
    plot(M, True)
