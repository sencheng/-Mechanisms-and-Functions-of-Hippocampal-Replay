# basic imports
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt


def compute_strength_map(C):
    strength_map = np.zeros((7, 11))
    for e, exp in enumerate(C):
        s = e % 77
        x = int(s/11)
        y = s - x * 11
        strength_map[x, y] += exp
        
    return strength_map

def plot_strengths(strengths: dict):
    plt.figure(1, figsize=(11, 14))
    plt.subplots_adjust(hspace=0.5)
    for i, condition in enumerate(strengths):
        plt.subplot(4, 2, i * 2 + 1)
        plt.text(11, 8.25, 'Condition %s' % condition, fontsize=20)
        plt.title('1st Half')
        plt.pcolor(np.flip(compute_strength_map(strengths[condition]['mid']), axis=0), cmap='hot')
        plt.xticks(np.array([0, 5, 10]) + 0.5, np.array([1, 6, 11]))
        plt.yticks(np.array([0, 2, 6]) + 0.5, np.array([1, 3, 7]))
        plt.colorbar()
        plt.subplot(4, 2, i * 2 + 2)
        plt.title('2nd Half')
        plt.pcolor(np.flip(compute_strength_map(strengths[condition]['end']), axis=0), cmap='hot')
        plt.xticks(np.array([0, 5, 10]) + 0.5, np.array([1, 6, 11]))
        plt.yticks(np.array([0, 2, 6]) + 0.5, np.array([1, 3, 7]))
        plt.colorbar()
    plt.savefig('plots/strengths.png', dpi=200, bbox_inches='tight', transparent=True)
    plt.savefig('plots/strengths.svg', dpi=200, bbox_inches='tight', transparent=True)

if __name__ == '__main__':
    # load simulation data
    strengths = pickle.load(open('data/strengths.pkl', 'rb'))
    
    # make sure that the directory for storing the simulation results exists
    os.makedirs('plots/', exist_ok=True)
    
    # plot strengths
    plot_strengths(strengths)
    