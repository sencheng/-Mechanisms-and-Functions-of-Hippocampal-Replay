# basic imports
import numpy as np
import pickle
import matplotlib.pyplot as plt


def count_pairs(replays):
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

def analyze(data):
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

def plot(consecutive_pairs_start, consecutive_pairs_end, fileName, title=''):
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
    plt.savefig(fileName + '.png', dpi=200, bbox_inches='tight', transparent=True)
    plt.savefig(fileName + '.svg', dpi=200, bbox_inches='tight', transparent=True)
    plt.close('all')
    
def plot_directionality(directionality, fileName, title=''):
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
    plt.savefig(fileName + '.png', dpi=200, bbox_inches='tight', transparent=True)
    plt.savefig(fileName + '.svg', dpi=200, bbox_inches='tight', transparent=True)
    plt.close('all')


# directionality linear track
directionality = {'start': {'default': None, 'reverse': None}, 'end': {'default': None, 'reverse': None}}
file_name = 'data/linear_track_mode_default_recency_N.pkl'
pairs_start, pairs_end = analyze(pickle.load(open(file_name, 'rb')))
directionality['start']['default'] = np.mean(np.array(pairs_start['forward']) - np.array(pairs_start['reverse']), axis=0)
directionality['end']['default'] = np.mean(np.array(pairs_end['forward']) - np.array(pairs_end['reverse']), axis=0)
file_name = 'data/linear_track_mode_reverse_recency_N.pkl'
pairs_start, pairs_end = analyze(pickle.load(open(file_name, 'rb')))
directionality['start']['reverse'] = np.mean(np.array(pairs_start['forward']) - np.array(pairs_start['reverse']), axis=0)
directionality['end']['reverse'] = np.mean(np.array(pairs_end['forward']) - np.array(pairs_end['reverse']), axis=0)
plot_directionality(directionality, 'plots/directionality_linear_track', title='Linear Track')

# directionality open field
directionality = {'start': {'default': None, 'reverse': None}, 'end': {'default': None, 'reverse': None}}
file_name = 'data/open_field_mode_default_recency_N.pkl'
pairs_start, pairs_end = analyze(pickle.load(open(file_name, 'rb')))
directionality['start']['default'] = np.mean(np.array(pairs_start['forward']) - np.array(pairs_start['reverse']), axis=0)
directionality['end']['default'] = np.mean(np.array(pairs_end['forward']) - np.array(pairs_end['reverse']), axis=0)
file_name = 'data/open_field_mode_reverse_recency_N.pkl'
pairs_start, pairs_end = analyze(pickle.load(open(file_name, 'rb')))
directionality['start']['reverse'] = np.mean(np.array(pairs_start['forward']) - np.array(pairs_start['reverse']), axis=0)
directionality['end']['reverse'] = np.mean(np.array(pairs_end['forward']) - np.array(pairs_end['reverse']), axis=0)
plot_directionality(directionality, 'plots/directionality_open_field', title='Open Field')

# directionality labyrinth
directionality = {'start': {'default': None, 'reverse': None}, 'end': {'default': None, 'reverse': None}}
file_name = 'data/labyrinth_mode_default_recency_N.pkl'
pairs_start, pairs_end = analyze(pickle.load(open(file_name, 'rb')))
directionality['start']['default'] = np.mean(np.array(pairs_start['forward']) - np.array(pairs_start['reverse']), axis=0)
directionality['end']['default'] = np.mean(np.array(pairs_end['forward']) - np.array(pairs_end['reverse']), axis=0)
file_name = 'data/labyrinth_mode_reverse_recency_N.pkl'
pairs_start, pairs_end = analyze(pickle.load(open(file_name, 'rb')))
directionality['start']['reverse'] = np.mean(np.array(pairs_start['forward']) - np.array(pairs_start['reverse']), axis=0)
directionality['end']['reverse'] = np.mean(np.array(pairs_end['forward']) - np.array(pairs_end['reverse']), axis=0)
plot_directionality(directionality, 'plots/directionality_labyrinth', title='Labyrinth')
                                  
fileName = 'data/linear_track_mode_reverse_recency_N.pkl'
data = pickle.load(open(fileName, 'rb'))
consecutive_pairs_start, consecutive_pairs_end = analyze(data)
plot(consecutive_pairs_start, consecutive_pairs_end, 'plots/linear_track_mode_reverse_recency_N', 'Linear Track, Reverse Mode')

fileName = 'data/linear_track_mode_reverse_recency_Y.pkl'
data = pickle.load(open(fileName, 'rb'))
consecutive_pairs_start, consecutive_pairs_end = analyze(data)
plot(consecutive_pairs_start, consecutive_pairs_end, 'plots/linear_track_mode_reverse_recency_Y', 'Linear Track, Reverse Mode, Recency')

fileName = 'data/linear_track_mode_default_recency_N.pkl'
data = pickle.load(open(fileName, 'rb'))
consecutive_pairs_start, consecutive_pairs_end = analyze(data)
plot(consecutive_pairs_start, consecutive_pairs_end, 'plots/linear_track_mode_default_recency_N', 'Linear Track, Default Mode')

fileName = 'data/linear_track_mode_default_recency_Y.pkl'
data = pickle.load(open(fileName, 'rb'))
consecutive_pairs_start, consecutive_pairs_end = analyze(data)
plot(consecutive_pairs_start, consecutive_pairs_end, 'plots/linear_track_mode_default_recency_Y', 'Linear Track, Default Mode, Recency')


fileName = 'data/open_field_mode_reverse_recency_N.pkl'
data = pickle.load(open(fileName, 'rb'))
consecutive_pairs_start, consecutive_pairs_end = analyze(data)
plot(consecutive_pairs_start, consecutive_pairs_end, 'plots/open_field_mode_reverse_recency_N', 'Open Field, Reverse Mode')

fileName = 'data/open_field_mode_reverse_recency_Y.pkl'
data = pickle.load(open(fileName, 'rb'))
consecutive_pairs_start, consecutive_pairs_end = analyze(data)
plot(consecutive_pairs_start, consecutive_pairs_end, 'plots/open_field_mode_reverse_recency_Y', 'Open Field, Reverse Mode, Recency')

fileName = 'data/open_field_mode_default_recency_N.pkl'
data = pickle.load(open(fileName, 'rb'))
consecutive_pairs_start, consecutive_pairs_end = analyze(data)
plot(consecutive_pairs_start, consecutive_pairs_end, 'plots/open_field_mode_default_recency_N', 'Open Field, Default Mode')

fileName = 'data/open_field_mode_default_recency_Y.pkl'
data = pickle.load(open(fileName, 'rb'))
consecutive_pairs_start, consecutive_pairs_end = analyze(data)
plot(consecutive_pairs_start, consecutive_pairs_end, 'plots/open_field_mode_default_recency_Y', 'Open Field, Default Mode, Recency')


fileName = 'data/labyrinth_mode_reverse_recency_N.pkl'
data = pickle.load(open(fileName, 'rb'))
consecutive_pairs_start, consecutive_pairs_end = analyze(data)
plot(consecutive_pairs_start, consecutive_pairs_end, 'plots/labyrinth_mode_reverse_recency_N', 'Labyrinth, Reverse Mode')

fileName = 'data/labyrinth_mode_reverse_recency_Y.pkl'
data = pickle.load(open(fileName, 'rb'))
consecutive_pairs_start, consecutive_pairs_end = analyze(data)
plot(consecutive_pairs_start, consecutive_pairs_end, 'plots/labyrinth_mode_reverse_recency_Y', 'Labyrinth, Reverse Mode, Recency')

fileName = 'data/labyrinth_mode_default_recency_N.pkl'
data = pickle.load(open(fileName, 'rb'))
consecutive_pairs_start, consecutive_pairs_end = analyze(data)
plot(consecutive_pairs_start, consecutive_pairs_end, 'plots/labyrinth_mode_default_recency_N', 'Labyrinth, Default Mode')

fileName = 'data/labyrinth_mode_default_recency_Y.pkl'
data = pickle.load(open(fileName, 'rb'))
consecutive_pairs_start, consecutive_pairs_end = analyze(data)
plot(consecutive_pairs_start, consecutive_pairs_end, 'plots/labyrinth_mode_default_recency_Y', 'Labyrinth, Default Mode, Recency')
