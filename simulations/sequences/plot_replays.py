# basic imports
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection


def generate_polygons_replay(replay: list, dims: tuple) -> np.ndarray:
    P = []
    for e, exp in enumerate(replay):
        a = exp['action']
        s = exp['state']
        x = int(s/dims[1])
        y = s - x * dims[1]
        coords = None
        if a == 0:
            coords = np.array([[1/3, 1/3], [1/2, 0.0], [2/3,1/3]]) + np.array([[x, y]])
        elif a == 1:
            coords = np.array([[0.0, 0.5], [1/3, 1/3], [1/3, 2/3]]) + np.array([[x, y]])
        elif a == 2:
            coords = np.array([[1/3, 2/3], [2/3, 2/3], [1/2, 1.]]) + np.array([[x, y]])
        elif a == 3:
            coords = np.array([[2/3, 1/3], [1., 1/2], [2/3, 2/3]]) + np.array([[x, y]])
        coords = np.flip(coords)
        #coords[1] = dims[0] - coords[1]
        P += [Polygon(coords)]
    P = PatchCollection(P)
    P.set_array(np.arange(len(replay)) + 1.)
    P.set_cmap('hot')
    P.set_clim([0, None])
    
    return P
        
def plot_replays(replays: tuple, dims: tuple, file_name: str):
    fig_dims = (5, 5 * dims[0]/10)
    for i, replay in enumerate(replays):
        plt.figure(1, figsize=fig_dims)
        plt.pcolor(np.zeros(dims), cmap='hot', vmin=0, vmax=1)
        for j in range(dims[1] - 1):
            plt.axvline(j+1, color='w', linewidth=0.5)
        for j in range(dims[0] - 1):
            plt.axhline(j+1, color='w', linewidth=0.5)
        plt.xlim(0, dims[1])
        plt.ylim(0, dims[0])
        ax = plt.gca()
        # generate polygons
        P = generate_polygons_replay(replay, dims)
        ax.add_collection(P)
        #plt.colorbar()
        plt.xticks(np.array([0, dims[1] - 1]) + 0.5,np.array([1, dims[1]]), fontsize=15)
        plt.yticks(np.array([0, dims[0] - 1]) + 0.5,np.array([1, dims[0]]), fontsize=15)
        plt.savefig('%s_%02d.png' % (file_name, i), dpi=200, bbox_inches='tight', transparent=True)
        plt.savefig('%s_%02d.svg' % (file_name, i), dpi=200, bbox_inches='tight', transparent=True)
        plt.close('all')

if __name__ == '__main__':
    # params
    environments = ['linear_track', 'open_field', 'labyrinth']
    environment_dims = {'linear_track': (1, 10), 'open_field': (10, 10), 'labyrinth': (10, 10)}
    modes = ['default', 'reverse']
    
    # make sure that the directories for storing the plots exists
    os.makedirs('plots/', exist_ok=True)
    for env in environments:
        os.makedirs('plots/%s/' % env, exist_ok=True)
    
    # plot replays
    for env in environments:
        for mode in modes:
            replays_start, replays_end = pickle.load(open('data/%s_mode_%s.pkl' % (env, mode), 'rb'))
            # trial begin
            plot_replays(replays_start[0][1:6], environment_dims[env], 'plots/%s/%s_start_early' % (env, mode))
            plot_replays(replays_start[0][-5:], environment_dims[env], 'plots/%s/%s_start_late' % (env, mode))
            # trial end
            plot_replays(replays_end[0][1:6], environment_dims[env], 'plots/%s/%s_end_early' % (env, mode))
            plot_replays(replays_end[0][-5:], environment_dims[env], 'plots/%s/%s_end_late' % (env, mode))
    