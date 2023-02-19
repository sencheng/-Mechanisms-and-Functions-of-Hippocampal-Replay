# basic imports
import numpy as np
import pickle
import matplotlib.pyplot as plt

    
def plot(data: dict, file_name: str):
    '''
    This function plots the learning performance of agents trained with SFMA and random replay.
    
    Parameters
    ----------
    data :                              Dictionary containing the learning performance for different agents.
    file_name :                         The name that the plot will be saved as.
    
    Returns
    ----------
    None
    '''
    plt.figure(1)
    plt.title('Learning Performance', fontsize=20)
    plt.plot(np.arange(200) + 1, np.mean(np.array(data['reverse']['rewards'] * 100), axis=0), color='g', linestyle='--', label='SFMA (Reverse)')
    plt.plot(np.arange(200) + 1, np.mean(np.array(data['default']['rewards'] * 100), axis=0), color='g', label='SFMA (Default)')
    plt.plot(np.arange(200) + 1, np.mean(np.array(data['random']['rewards'] * 100), axis=0), color='k', label='Random Replay')
    plt.xticks(np.array([1, 25, 50, 75, 100, 125, 150, 175, 200]), np.array([1, 25, 50, 75, 100, 125, 150, 175, 200]), fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlim(1, 200)
    plt.ylim(0, 1)
    plt.xlabel('Trial', fontsize=15)
    plt.ylabel('Correct Choice [%]', fontsize=15)
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    art = []
    lgd = plt.legend(loc=9, bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize=15, framealpha=0.)
    art.append(lgd)
    plt.savefig(file_name + '.png', dpi=200, bbox_inches='tight', transparent=True)
    plt.savefig(file_name + '.svg', dpi=200, bbox_inches='tight', transparent=True)
    plt.close('all')    
            
    
sfma = pickle.load(open('data/sfma_learning.pkl', 'rb'))
random = pickle.load(open('data/random_learning.pkl', 'rb'))


for recency in [True, False]:
    data = {'default': sfma['default'][recency], 'reverse': sfma['reverse'][recency], 'random': random}
    plot(data, 'plots/learning_recency_' + ('Y' if recency else 'N'))
