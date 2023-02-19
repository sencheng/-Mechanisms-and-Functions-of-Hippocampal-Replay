# basic imports
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection


def generate_polygons(DR):
    P = []
    for e, exp in enumerate(DR):
        a = int(e/25)
        s = e % 25
        x = int(s/5)
        y = s - x * 5
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
        P += [Polygon(coords)]
    P = PatchCollection(P)
    P.set_array(DR)
    P.set_cmap('hot')
    P.set_clim([0, None])
    
    return P

def generate_polygons_replay(replay):
    P = []
    for e, exp in enumerate(replay):
        a = exp['action']
        s = exp['state']
        x = int(s/5)
        y = s - x * 5
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
        P += [Polygon(coords)]
    P = PatchCollection(P)
    P.set_array(np.arange(len(replay)) + 1.)
    P.set_cmap('hot')
    P.set_clim([0, None])
    
    return P

def plot_variables(data):
    titles = {'C': '$C(e)$', 'D': '$D(e, e_%d)$', 'I': '$I(e)$', 'R': '$R(e|e_%d)$', 'P': '$P(e|e_%d)$'}
    for var in data:
        for s, step in enumerate(data[var]):
            title = titles[var]
            v_max = 1.
            if var in ['D', 'R', 'P']:
                title = title % (s-1)
            if var == 'D':
                v_max = None
            #similarity_map = np.reshape(DR, (11, 11))
            plt.figure(1, figsize=(5, 5))
            plt.title(title, fontsize=20)
            plt.pcolor(np.zeros((5, 5)), cmap='hot', vmin=0, vmax=v_max)
            for j in range(4):
                plt.axvline(j+1, color='w', linewidth=0.5)
                plt.axhline(j+1, color='w', linewidth=0.5)
            plt.xlim(0, 5)
            plt.ylim(0, 5)
            ax = plt.gca()
            # generate polygons
            P = generate_polygons(step)
            ax.add_collection(P)
            #plt.colorbar()
            plt.xticks(np.array([0, 2, 4]) + 0.5,np.array([1, 3, 5]), fontsize=15)
            plt.yticks(np.array([0, 2, 4]) + 0.5,np.array([1, 3, 5]), fontsize=15)
            plt.savefig('plots/%s/%s_%02d.png' % (var, var, s), dpi=200, bbox_inches='tight', transparent=True)
            plt.savefig('plots/%s/%s_%02d.svg' % (var, var, s), dpi=200, bbox_inches='tight', transparent=True)
            plt.close('all')
        
def plot_replays(replay):
    for i in range(len(replay)):
        #similarity_map = np.reshape(DR, (11, 11))
        plt.figure(1, figsize=(5, 5))
        plt.title('$e_%d$' % i, fontsize=20)
        plt.pcolor(np.zeros((5, 5)), cmap='hot', vmin=0, vmax=1)
        for j in range(4):
            plt.axvline(j+1, color='w', linewidth=0.5)
            plt.axhline(j+1, color='w', linewidth=0.5)
        plt.xlim(0, 5)
        plt.ylim(0, 5)
        ax = plt.gca()
        # generate polygons
        P = generate_polygons_replay(replay[:i+1])
        ax.add_collection(P)
        #
        #plt.colorbar()
        plt.xticks(np.array([0, 2, 4]) + 0.5,np.array([1, 3, 5]), fontsize=15)
        plt.yticks(np.array([0, 2, 4]) + 0.5,np.array([1, 3, 5]), fontsize=15)
        plt.savefig('plots/replay_%02d.png' % i, dpi=200, bbox_inches='tight', transparent=True)
        plt.savefig('plots/replay_%02d.svg' % i, dpi=200, bbox_inches='tight', transparent=True)
        plt.close('all')

if __name__ == '__main__':
    # load simulation data
    replay = pickle.load(open('data/replay.pkl', 'rb'))
    var_trace = pickle.load(open('data/var_trace.pkl', 'rb'))
    
    # make sure that the directories for storing the plots exists
    os.makedirs('plots/', exist_ok=True)
    for var in var_trace:
        os.makedirs('plots/%s/' % var, exist_ok=True)
    
    # plot similarities
    plot_variables(var_trace)
    plot_replays(replay)
    