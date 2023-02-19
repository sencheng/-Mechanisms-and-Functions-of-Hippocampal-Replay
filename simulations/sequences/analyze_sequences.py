# basic imports
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt


def count_pairs(replays: list) -> (list, list, list):
    '''
    This function counts the number of forward, reverse and unordered replay pairs.
    
    Parameters
    ----------
    replays :                           A list of replays.
    
    Returns
    ----------
    forward_pairs :                     A list containing the number of forward pairs found in each replay.
    reverse_pairs :                     A list containing the number of reverse pairs found in each replay.
    unordered_pairs :                   A list containing the number of unordered pairs found in each replay.
    '''
    fwd, rev, unordered = [], [], []
    for replay in replays:
        pairs_fwd, pairs_rev, pairs_unordered = 0, 0, 0
        for s, step in enumerate(replay[1:]):
            if step['state'] == replay[s]['next_state']:
                pairs_fwd += 1
            elif step['next_state'] == replay[s]['state']:
                pairs_rev += 1
            else:
                pairs_unordered += 1
        fwd += [pairs_fwd]
        rev += [pairs_rev]
        unordered += [pairs_unordered]
    
    return fwd, rev, unordered

def count_sequences(replays: list, max_length: int) -> np.ndarray:
    '''
    This function computes the sequence length histogram for each trial.
    
    Parameters
    ----------
    replays :                           A list of replays.
    max_length :                        The maximum replay length.
    
    Returns
    ----------
    histograms :                        A numpy array containing the sequence length histogram for each trial.
    '''
    histograms = []
    for replay in replays:
        histogram = np.zeros(max_length)
        if len(replay) > 0:
            prev_type = None
            state, next_state = replay[0]['state'], replay[0]['next_state']
            length = 1
            for exp in replay[1:]:
                # determine directionality of current consecutive pair
                current_type = None
                if next_state == exp['state']:
                    current_type = 'forward'
                elif state == exp['next_state']:
                    current_type = 'reverse'
                else:
                    current_type = 'unordered'
                # check if fwd/rev sequence continues
                if prev_type == current_type or prev_type == 'unordered':
                    length += 1
                elif prev_type is not None and prev_type != current_type:
                    histogram[length] += 1
                    length = 1
                # update values for previous experience
                prev_type = current_type
                state, next_state = exp['state'], exp['next_state']
            histogram[length] += 1
        histograms.append(histogram)
            
    return np.array(histograms)

def analyze(data: list) -> (dict, dict):
    '''
    This function analyzes the number of different replay pairs for replays occuring at the begin and end of a trial.
    
    Parameters
    ----------
    data :                              A list containing replays occuring at the begin and end of trials.
    
    Returns
    ----------
    pairs_start :                       A dictionary containing the different types of replays pairs occuring at the begin of a trial.
    pairs_end :                         A dictionary containing the different types of replays pairs occuring at the end of a trial.
    '''
    replays_start, replays_end = data
    consecutive_pairs_start = {'forward': [], 'reverse': [], 'unordered': []}
    consecutive_pairs_end = {'forward': [], 'reverse': [], 'unordered': []}
    for run in range(len(replays_start)):
        # count pairs for replays at start
        fwd, rev, unordered = count_pairs(replays_start[run])
        consecutive_pairs_start['forward'] += [fwd]
        consecutive_pairs_start['reverse'] += [rev]
        consecutive_pairs_start['unordered'] += [unordered]
        # count pairs for replays at end
        fwd, rev, unordered = count_pairs(replays_end[run])
        consecutive_pairs_end['forward'] += [fwd]
        consecutive_pairs_end['reverse'] += [rev]
        consecutive_pairs_end['unordered'] += [unordered]
        
    return consecutive_pairs_start, consecutive_pairs_end

def analyze_sequences(data: list, max_length: int) -> (list, list):
    '''
    This function computes the sequence length histograms for trial start/end replays for all runs.
    
    Parameters
    ----------
    data :                              The trial begin and trial end replay lists.
    max_length :                        The maximum replay length.
    
    Returns
    ----------
    histograms_start :                  A list containing the sequence length histograms for trial begin replays for all runs.
    histograms_end :                    A list containing the sequence length histograms for trial end replays for all runs.
    '''
    replays_start, replays_end = data
    histograms_start = []
    histograms_end = []
    for run in range(len(replays_start)):
        histograms_start.append(count_sequences(replays_start[run], max_length))
        histograms_end.append(count_sequences(replays_end[run], max_length))
    
    return histograms_start, histograms_end

def plot(consecutive_pairs_start: dict, consecutive_pairs_end: dict, file_name: str, title: str = ''):
    '''
    This function plots the number of different replay pairs for replays occuring at the begin and end of a trial.
    
    Parameters
    ----------
    consecutive_pairs_start :           A dictionary containing the different types of replays pairs occuring at the begin of a trial.
    consecutive_pairs_end :             A dictionary containing the different types of replays pairs occuring at the end of a trial.
    file_name :                         The name that the plot will be saved as.
    title :                             A plot's title.
    
    Returns
    ----------
    None
    '''
    colors = {'forward': 'b', 'reverse': 'r', 'unordered': 'g'}
    plt.figure(1)
    plt.suptitle(title, fontsize=15, position=(0.5, 1.03))
    plt.subplots_adjust(hspace=0.8)
    plt.subplot(2, 1, 1)
    plt.title('Trial Start Replays')
    for pair in consecutive_pairs_start:
        plt.plot(np.mean(np.array(consecutive_pairs_start[pair]), axis=0), color=colors[pair])
    plt.xlabel('Trial')
    plt.ylabel('Pairs [#]')
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.subplot(2, 1, 2)
    plt.title('Trial End Replays')
    for pair in consecutive_pairs_end:
        plt.plot(np.mean(np.array(consecutive_pairs_end[pair]), axis=0), color=colors[pair], label=pair)
    plt.xlabel('Trial')
    plt.ylabel('Pairs [#]')
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    art = []
    lgd = plt.legend(loc=9, bbox_to_anchor=(0.5, -0.5), ncol=3, fontsize=12, framealpha=0.)
    art.append(lgd)
    plt.savefig(file_name + '.png', dpi=200, bbox_inches='tight', transparent=True)
    plt.savefig(file_name + '.svg', dpi=200, bbox_inches='tight', transparent=True)
    plt.close('all')
    
def plot_directionality(directionality: dict, file_name: str, title: str = ''):
    '''
    This function plots the overall directionality of replays occuring at the begin and end of a trial.
    
    Parameters
    ----------
    directionality :                    A dictionary containing the overall directionality of replay.
    file_name :                         The name that the plot will be saved as.
    title :                             A plot's title.
    
    Returns
    ----------
    None
    '''
    plt.figure(1)
    plt.suptitle(title, fontsize=15, position=(0.5, 1.03))
    plt.subplots_adjust(hspace=0.8)
    plt.subplot(2, 1, 1)
    plt.title('Trial Start Replays')
    max_magnitude = max(np.amax(directionality['start']['default']), np.amax(directionality['start']['reverse'])) * 1.05
    min_magnitude = min(np.amin(directionality['start']['default']), np.amin(directionality['start']['reverse'])) * 1.05
    idx = np.arange(directionality['start']['default'].shape[0]) + 1
    plt.plot(idx, directionality['start']['default'], color='g', label='SFMA (Default)', zorder=10)
    plt.plot(idx, directionality['start']['reverse'], color='g', linestyle='-.', label='SFMA (Reverse)', zorder=10)
    plt.axhline(0, color='grey', zorder=1)
    plt.xlabel('Trial')
    plt.ylabel('Directionality\n[$pairs_{fwd}-pairs_{bwd}$]')
    plt.xlim(np.amin(idx), np.amax(idx))
    plt.ylim(min_magnitude, max_magnitude)
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.subplot(2, 1, 2)
    plt.title('Trial End Replays')
    max_magnitude = max(np.amax(directionality['end']['default']), np.amax(directionality['end']['reverse'])) * 1.05
    min_magnitude = min(np.amin(directionality['end']['default']), np.amin(directionality['end']['reverse'])) * 1.05
    plt.plot(idx, directionality['end']['default'], color='g', label='SFMA (Default)', zorder=10)
    plt.plot(idx, directionality['end']['reverse'], color='g', linestyle='-.', label='SFMA (Reverse)', zorder=10)
    plt.axhline(0, color='grey', zorder=1)
    plt.xlabel('Trial')
    plt.ylabel('Directionality\n[$pairs_{fwd}-pairs_{bwd}$]')
    plt.xlim(np.amin(idx), np.amax(idx))
    plt.ylim(min_magnitude, max_magnitude)
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    art = []
    lgd = plt.legend(loc=9, bbox_to_anchor=(0.5, -0.5), ncol=3, fontsize=12, framealpha=0.)
    art.append(lgd)
    plt.savefig(file_name + '.png', dpi=200, bbox_inches='tight', transparent=True)
    plt.savefig(file_name + '.svg', dpi=200, bbox_inches='tight', transparent=True)
    plt.close('all')
    
def plot_sequences(sequences: dict):
    '''
    This function plots the trial sequence lengths for the different environments.
    
    Parameters
    ----------
    sequences :                         A dictionary containing the sequence length histograms for trial begin and trial end replays for all trials.
    
    Returns
    ----------
    None
    '''
    titles = {'linear_track': 'Linear Track', 'open_field': 'Open Field', 'labyrinth': 'Labyrinth'}
    plt.figure(1, figsize=(12, 3))
    for i, env in enumerate(sequences):
        plt.subplot(1, 3, i + 1)
        # prepare data
        sequences_start = np.zeros(100)
        sequences_end = np.zeros(100)
        for run in range(len(sequences[env]['start'])):
            for t, hist in enumerate(sequences[env]['start'][run]):
                sequences_start[t] += np.sum(hist[2:])
        for run in range(len(sequences[env]['end'])):
            for t, hist in enumerate(sequences[env]['end'][run]):
                sequences_end[t] += np.sum(hist[2:])
        sequences_start /= 100
        sequences_end /= 100
        plt.plot(np.arange(99) + 2, sequences_start[1:], label='Trial Begin')
        plt.plot(np.arange(100) + 1, sequences_end, label='Trial End')
        plt.title(titles[env])
        plt.xlabel('Trial')
        plt.ylabel('Sequences [#]')
        plt.xlim(1, 100)
        plt.ylim(0.95, None)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        if i == 1:
            art = []
            lgd = plt.legend(loc=9, bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=12, framealpha=0.)
            art.append(lgd)
    plt.savefig('plots/sequence_lengths.png', dpi=200, bbox_inches='tight', transparent=True)
    plt.savefig('plots/sequence_lengths.svg', dpi=200, bbox_inches='tight', transparent=True)

if __name__ == '__main__':
    # params
    environments = ['linear_track', 'open_field', 'labyrinth']
    modes = ['default', 'reverse']
    
    # define plot titles
    env_titles = {'linear_track': 'Linear Track', 'open_field': 'Open Field', 'labyrinth': 'Labyrinth'}
    mode_titles = {'default': 'Default Mode', 'reverse': 'Reverse Mode'}
    
    # make sure that the directory for storing the simulation results exists
    os.makedirs('plots/', exist_ok=True)
    
    # plot directionality
    for env in environments:
        directionality = {'start': {'default': None, 'reverse': None}, 'end': {'default': None, 'reverse': None}}
        # directionality for default mode
        file_name = 'data/%s_mode_default.pkl' % env
        pairs_start, pairs_end = analyze(pickle.load(open(file_name, 'rb')))
        directionality['start']['default'] = np.mean(np.array(pairs_start['forward']) - np.array(pairs_start['reverse']), axis=0)
        directionality['end']['default'] = np.mean(np.array(pairs_end['forward']) - np.array(pairs_end['reverse']), axis=0)
        # directionality for reverse mode
        file_name = 'data/%s_mode_reverse.pkl' % env
        pairs_start, pairs_end = analyze(pickle.load(open(file_name, 'rb')))
        directionality['start']['reverse'] = np.mean(np.array(pairs_start['forward']) - np.array(pairs_start['reverse']), axis=0)
        directionality['end']['reverse'] = np.mean(np.array(pairs_end['forward']) - np.array(pairs_end['reverse']), axis=0)
        # plot directionality
        plot_directionality(directionality, 'plots/directionality_%s' % env, title=env_titles[env])
        
    # plot forward, reverse and unordered pairs
    for env in environments:
        for mode in modes:
            file_name = 'data/%s_mode_%s.pkl' % (env, mode)
            data = pickle.load(open(file_name, 'rb'))
            consecutive_pairs_start, consecutive_pairs_end = analyze(data)
            plot(consecutive_pairs_start, consecutive_pairs_end, 'plots/%s_mode_%s' % (env, mode), '%s, %s' % (env_titles[env], mode_titles[mode]))
    
    # plot sequences lengths
    for mode in ['default']:
        sequences = {}
        for env in environments:
            file_name = 'data/%s_mode_%s.pkl' % (env, mode)
            data = pickle.load(open(file_name, 'rb'))
            hist_start, hist_end = analyze_sequences(data, 100)
            sequences[env] = {'start': hist_start, 'end': hist_end}
        plot_sequences(sequences)
        