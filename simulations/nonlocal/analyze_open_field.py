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

def analyze_nonlocal(data: dict, width: int) -> dict:
    '''
    This function analyzes the simulation data.
    
    Parameters
    ----------
    data :                              The simulation data.
    width :                             The width of the environment.
    
    Returns
    ----------
    results :                           A dictionary containing the analyzed data.
    '''
    results = {'current_positions': [], 'replay_positions': [], 'distances': []}
    for current_state, replay_states in zip(data['current_states'], data['replay_states']):
        current_position = recover_position(current_state, width)
        replay_positions = [recover_position(state, width) for state in replay_states]
        results['current_positions'].append(current_position)
        results['replay_positions'].append(replay_positions)
        results['distances'].append(np.sort(np.sqrt(np.sum((replay_positions - current_position)**2, axis=1))))
            
    return results

def plot_heatmap(data: dict, weighting: int):
    '''
    This function plots the replay starting locations as a heatmap.
    
    Parameters
    ----------
    data :                              The analyzed simulation data.
    weighting :                         The influence of the current location.
    
    Returns
    ----------
    None
    '''
    # prepare heatmap
    fraction = np.zeros((10, 10))
    for trial in data['replay_positions']:
        for position in trial:
            fraction[position[0], position[1]] += 1
    fraction /= np.sum(fraction)
    # plot heatmap
    plt.figure(1, figsize=(4, 3.5))
    plt.title('Replay Starting Loactions\nWeighting = %d' % weighting)
    plt.pcolor(fraction, cmap='hot', vmin=0, vmax=None)
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()
    plt.savefig('plots/open_field_heatmap_%d.png' % weighting, dpi=200, bbox_inches='tight', transparent=True)
    plt.savefig('plots/open_field_heatmap_%d.svg' % weighting, dpi=200, bbox_inches='tight', transparent=True)
    plt.close('all')

def plot_histogram(data: dict, weighting: int):
    '''
    This function plots a histogram of replay start location distances.
    
    Parameters
    ----------
    data :                              The analyzed simulation data.
    weighting :                         The influence of the current location.
    
    Returns
    ----------
    None
    '''
    # prepare distances
    distances = np.sort(np.array(data['distances']).flatten())
    counts, bins = np.histogram(distances, 11, (0, 11))
    counts, bins = np.zeros(15), np.arange(15)
    for i in range(15):
        counts[i] = np.sum((distances >= i) * (distances < (i + 1)))
    # plot histogram
    plt.figure(1)
    plt.title('Replay Starting Location Distances\nWeighting = %d' % weighting)
    plt.xlabel('Distance to Current Location')
    plt.ylabel('Occurrence')
    #plt.stairs(counts, bins)
    plt.bar(bins, counts)
    plt.xlim(-0.5, 10.5)
    plt.ylim(0, 1400)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.savefig('plots/open_field_histogram_%d.png' % weighting, dpi=200, bbox_inches='tight', transparent=True)
    plt.savefig('plots/open_field_histogram_%d.svg' % weighting, dpi=200, bbox_inches='tight', transparent=True)
    plt.close('all')
        
def plot_histogram_adjusted(data: dict, weighting: int):
    '''
    This function plots a histogram of replay start location distances.
    Raw bin counts are divided by the theoretical max bin counts.
    
    Parameters
    ----------
    data :                              The analyzed simulation data.
    weighting :                         The influence of the current location.
    
    Returns
    ----------
    None
    '''
    positions = []
    for i in range(7):
        for j in range(11):
            positions.append([i, j])
    dists = np.sqrt(np.sum((positions - np.array([3, 0]))**2, axis=1))
    max_counts = np.zeros(15)
    for i in range(15):
        max_counts[i] = np.sum((dists >= i) * (dists < (i + 1))) * 2000
    max_counts[max_counts==0] = 1
    # prepare distances
    distances = np.sort(np.array(data['distances']).flatten())
    counts, bins = np.histogram(distances, 11, (0, 11))
    counts, bins = np.zeros(15), np.arange(15)
    for i in range(15):
        counts[i] = np.sum((distances >= i) * (distances < (i + 1)))
    # plot histogram
    plt.figure(1)
    plt.title('Replay Starting Location Distances\nWeighting = %d' % weighting)
    plt.xlabel('Distance to Current Location')
    plt.ylabel('Adjusted Occurrence')
    #plt.stairs(counts, bins)
    plt.bar(bins, counts/max_counts)
    plt.xlim(-0.5, 10.5)
    plt.ylim(0, 0.3)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.savefig('plots/open_field_histogram_adjusted_%d.png' % weighting, dpi=200, bbox_inches='tight', transparent=True)
    plt.savefig('plots/open_field_histogram_adjusted_%d.svg' % weighting, dpi=200, bbox_inches='tight', transparent=True)
    plt.close('all')
        

if __name__ == '__main__':
    # params
    weightings_local = [0, 5, 10]
    width = 10
    # load and analyze data
    results = {weighting: analyze_nonlocal(pickle.load(open('data/results_open_field_%d.pkl' % weighting, 'rb')), width) for weighting in weightings_local}
    # make sure that the directory for plotting the simulation results exists
    os.makedirs('plots/', exist_ok=True)
    # plot results
    for weighting in results:
        plot_heatmap(results[weighting], weighting)
        #plot_histogram(results[weighting], weighting)
        plot_histogram_adjusted(results[weighting], weighting)
    