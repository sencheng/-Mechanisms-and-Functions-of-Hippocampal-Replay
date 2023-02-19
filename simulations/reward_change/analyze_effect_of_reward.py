# basic imports
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt


def prepare_data(data: dict) -> (dict, dict):
    '''
    This function averages and prepares escape latency and reverse mode probabilities.
    
    Parameters
    ----------
    data :                              A dictionary containing the simulation data for open field and T-maze environments.
    
    Returns
    ----------
    escape_latency :                    A dictionary containing the average trial escape latencies for open field and T-maze environments.
    mode :                              A dictionary containing the trial reverse mode probabilities for open field and T-maze environments.
    '''
    escape_latency, mode = {}, {}
    for world in data:
        escape_latency[world] = np.mean(np.array(data[world]['escape_latency']), axis=0)
        mode[world] = np.mean(np.array(data[world]['mode']), axis=0) * 100
    
    return escape_latency, mode
    

def plot_escape_latency(data: dict):
    '''
    This function plots the learning performance in open field and T-maze environments.
    
    Parameters
    ----------
    data :                              A dictionary containing the average trial escape latencies for open field and T-maze environments.
    
    Returns
    ----------
    None
    '''
    plt.figure(1, figsize=(12, 4))
    # open field
    plt.subplot(1, 2, 1)
    plt.title('Open Field')
    plt.plot(np.arange(300) + 1, data['open_field'])
    plt.axvline(100, linestyle='--', color='k')
    plt.axvline(200, linestyle='--', color='k')
    plt.xlabel('Trial')
    plt.ylabel('Escape Latency [#steps]')
    plt.xlim(0.5, 300.5)
    plt.ylim(0, 100)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    # T-maze
    plt.subplot(1, 2, 2)
    plt.title('T-Maze')
    plt.plot(np.arange(300) + 1, data['t_maze'])
    plt.axvline(100, linestyle='--', color='k')
    plt.axvline(200, linestyle='--', color='k')
    plt.xlabel('Trial')
    plt.ylabel('Escape Latency [#steps]')
    plt.xlim(0.5, 300.5)
    plt.ylim(0, 100)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    # save figure
    plt.savefig('plots/eor_learning.png', dpi=200, bbox_inches='tight', transparent=True)
    plt.savefig('plots/eor_learning.svg', dpi=200, bbox_inches='tight', transparent=True)
    plt.close('all')
    
def plot_mode_probability(data: dict):
    '''
    This function plots the probability of the reverse mode in open field and T-maze environments.
    
    Parameters
    ----------
    data :                              A dictionary containing the trial reverse mode probabilities for open field and T-maze environments.
    
    Returns
    ----------
    None
    '''
    plt.figure(1, figsize=(12, 4))
    # open field
    plt.subplot(1, 2, 1)
    plt.title('Open Field')
    plt.plot(np.arange(300) + 1, data['open_field'])
    plt.axvline(100, linestyle='--', color='k')
    plt.axvline(200, linestyle='--', color='k')
    plt.xlabel('Trial')
    plt.ylabel('Reverse Mode [%]')
    plt.xlim(0.5, 300.5)
    plt.ylim(0, 100)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    # T-maze
    plt.subplot(1, 2, 2)
    plt.title('T-Maze')
    plt.plot(np.arange(300) + 1, data['t_maze'])
    plt.axvline(100, linestyle='--', color='k')
    plt.axvline(200, linestyle='--', color='k')
    plt.xlabel('Trial')
    plt.ylabel('Reverse Mode [%]')
    plt.xlim(0.5, 300.5)
    plt.ylim(0, 100)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    # save figure
    plt.savefig('plots/eor_mode.png', dpi=200, bbox_inches='tight', transparent=True)
    plt.savefig('plots/eor_mode.svg', dpi=200, bbox_inches='tight', transparent=True)
    plt.close('all')
    
def plot_difference(data: dict):
    '''
    This function plots the difference in reverse mode probability after reward changes.
    
    Parameters
    ----------
    data :                              A dictionary containing the trial reverse mode probabilities for open field and T-maze environments.
    
    Returns
    ----------
    None
    '''
    change_open_field = np.array([np.mean(data['open_field'][100:105])/np.mean(data['open_field'][95:100]),
                                  np.mean(data['open_field'][200:205])/np.mean(data['open_field'][195:200])])
    change_t_maze = np.array([np.mean(data['t_maze'][100:105])/np.mean(data['t_maze'][95:100]),
                              np.mean(data['t_maze'][200:205])/np.mean(data['t_maze'][195:200])])
    plt.figure(1, figsize=(8, 4))
    # open field
    plt.subplot(1, 2, 1)
    plt.title('Open Field')
    plt.bar(np.arange(2), change_open_field * 100)
    plt.xlabel('Reward Change')
    plt.ylabel('Reverse Mode Probability Increase [%]')
    plt.xticks(np.arange(2), ['+0.1', '-0.1'])
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    # T-maze
    plt.subplot(1, 2, 2)
    plt.title('T-Maze')
    plt.bar(np.arange(2), change_t_maze * 100)
    plt.xlabel('Reward Change')
    plt.ylabel('Reverse Mode Probability Increase [%]')
    plt.xticks(np.arange(2), ['+0.1', '-0.1'])
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    # save figure
    plt.savefig('plots/eor_diff.png', dpi=200, bbox_inches='tight', transparent=True)
    plt.savefig('plots/eor_diff.svg', dpi=200, bbox_inches='tight', transparent=True)
    plt.close('all')

if __name__ == '__main__':
    # params
    worlds = ['open_field', 't_maze']
    
    # make sure that the directory for storing the simulation results exists
    os.makedirs('plots/', exist_ok=True)
    
    # prepare data
    escape_latency, mode = prepare_data({world: pickle.load(open('data/%s.pkl' % world, 'rb')) for world in worlds})
    # plot data
    plot_escape_latency(escape_latency)
    plot_mode_probability(mode)
    plot_difference(mode)
    