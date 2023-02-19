# basic imports
import numpy as np
import matplotlib.pyplot as plt
import pickle


def plot_replays(replays: list, file_name: str):
    '''
    This function plots replays (the first twelve).
    
    Parameters
    ----------
    replays :                           A containing the generated replays.
    file_name :                         The name that the file will be saved as.
    
    Returns
    ----------
    None
    '''
    plt.figure(1, figsize=(12, 4))
    for r, replay in enumerate(replays[:12]):
        plt.subplot(2, 6, r + 1)
        reactivation_map = np.zeros((20, 20))
        for s, state in enumerate(replay):
            reactivation_map[state[0], state[1]] = s
        reactivation_map /= s
        plt.pcolor(reactivation_map, cmap='hot', vmin=0, vmax=1)
        plt.xticks([])
        plt.yticks([])
    plt.savefig(file_name + '.png', dpi=200, bbox_inches='tight', transparent=True)
    plt.savefig(file_name + '.svg', dpi=200, bbox_inches='tight', transparent=True)
    plt.close('all')


if __name__ == '__main__':
    # load and plot replays
    replays = pickle.load(open('data/replays_original.pkl', 'rb'))
    plot_replays(replays, 'pma_original')
    replays = pickle.load(open('data/replays_adjusted.pkl', 'rb'))
    plot_replays(replays, 'pma_adjusted')
