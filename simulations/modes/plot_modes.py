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
        a = int(e/121)
        s = e % 121
        x = int(s/11)
        y = s - x * 11
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
    P.set_array(np.log(DR) - np.amin(np.log(DR)))
    P.set_cmap('hot')
    P.set_clim([21, None])
    
    return P

def generate_polygons_replay(replay):
    P = []
    for e, exp in enumerate(replay):
        a = exp['action']
        s = exp['state']
        x = int(s/11)
        y = s - x * 11
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

def plot_similarity(data):
    titles = {'default': 'Default', 'reverse': 'Reverse', 'forward': 'Forward', 'sweeping': 'Attractor'}
    for mode in data:
        DR = data[mode]['DR']
        #similarity_map = np.reshape(DR, (11, 11))
        plt.figure(1, figsize=(5, 5))
        plt.title(titles[mode], fontsize=20)
        plt.pcolor(np.zeros((11, 11)), cmap='hot', vmin=0, vmax=1)
        for j in range(10):
            plt.axvline(j+1, color='w', linewidth=0.5)
            plt.axhline(j+1, color='w', linewidth=0.5)
        plt.xlim(0, 11)
        plt.ylim(0, 11)
        ax = plt.gca()
        # generate polygons
        P = generate_polygons(DR)
        ax.add_collection(P)
        # mark experience
        plt.plot(np.array([2/3, 1, 2/3, 2/3]) + 5, np.array([1/3, 1/2, 2/3, 1/3]) + 5, color='springgreen')
        #plt.colorbar()
        plt.xticks(np.array([0, 5, 10]) + 0.5,np.array([1, 6, 11]), fontsize=15)
        plt.yticks(np.array([0, 5, 10]) + 0.5,np.array([1, 6, 11]), fontsize=15)
        plt.savefig('plots/DR_%s.png' % mode, dpi=200, bbox_inches='tight', transparent=True)
        plt.savefig('plots/DR_%s.svg' % mode, dpi=200, bbox_inches='tight', transparent=True)
        plt.close('all')
        
def plot_replays(data):
    for mode in data:
        for i, replay in enumerate(data[mode]['replay']):
            #similarity_map = np.reshape(DR, (11, 11))
            plt.figure(1, figsize=(5, 5))
            plt.pcolor(np.zeros((11, 11)), cmap='hot', vmin=0, vmax=1)
            for j in range(10):
                plt.axvline(j+1, color='w', linewidth=0.5)
                plt.axhline(j+1, color='w', linewidth=0.5)
            plt.xlim(0, 11)
            plt.ylim(0, 11)
            ax = plt.gca()
            # generate polygons
            P = generate_polygons_replay(replay)
            ax.add_collection(P)
            #
            #plt.colorbar()
            plt.xticks(np.array([0, 5, 10]) + 0.5,np.array([1, 6, 11]), fontsize=15)
            plt.yticks(np.array([0, 5, 10]) + 0.5,np.array([1, 6, 11]), fontsize=15)
            plt.savefig('plots/%s/replay_%02d.png' % (mode, i), dpi=200, bbox_inches='tight', transparent=True)
            plt.savefig('plots/%s/replay_%02d.svg' % (mode, i), dpi=200, bbox_inches='tight', transparent=True)
            plt.close('all')

if __name__ == '__main__':
    # load simulation data
    data = pickle.load(open('data/results.pkl', 'rb'))
    
    # make sure that the directories for storing the plots exists
    os.makedirs('plots/', exist_ok=True)
    for mode in data:
        os.makedirs('plots/%s/' % mode, exist_ok=True)
    
    # plot similarities
    plot_similarity(data)
    plot_replays(data)
    