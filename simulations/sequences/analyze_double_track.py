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

if __name__ == '__main__':
    # params
    environments = ['double_track']
    modes = ['default', 'reverse']
    
    # define plot titles
    env_titles = {'double_track': 'Double Track'}
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
    