# basic imports
import numpy as np
import pickle
import matplotlib.pyplot as plt


def load_and_prepare(prefix, modes, use_recency):
    analyzed_data = pickle.load(open('analysis/' + prefix + 'shortCuts_static.pkl', 'rb'))
    conditions = ['AA', 'RL', 'RA', 'AL']
    betas = [5]
    betas = np.linspace(5, 15, 11)
    center, sc, nsc = {}, {}, {}
    for mode in modes:
        center[mode], sc[mode], nsc[mode] = {}, {}, {}
        for recency in use_recency:
            center[mode][recency], sc[mode][recency], nsc[mode][recency] = {}, {}, {}
            for condition in conditions:
                center[mode][recency][condition], sc[mode][recency][condition], nsc[mode][recency][condition] = {}, {}, {}
                for beta in betas:
                    center[mode][recency][condition][beta] = analyzed_data[mode][recency][beta][condition]['center']
                    sc[mode][recency][condition][beta] = analyzed_data[mode][recency][beta][condition]['shortcuts']
                    nsc[mode][recency][condition][beta] = analyzed_data[mode][recency][beta][condition]['non_shortcuts']
    
    return center, sc, nsc

def plot_details(data, file_name):
    plt.figure(1, figsize=(12, 10))
    plt.subplots_adjust(hspace=0.9)
    plt.suptitle('Shortcut Replays Across Trials', fontsize=25)
    for c, condition in enumerate(data):
        plt.subplot(4, 1, c + 1)
        colors = plt.cm.jet(np.linspace(0, 1, len(data[condition])))
        for b, beta in enumerate(data[condition]):
            plt.title('Condition: ' + condition, fontsize=20)
            plt.plot(np.arange(11) + 10, data[condition][beta], label='$\\beta_M = ' + str(beta) + '$', color=colors[b])
            #plt.ylim(0, np.amax(data[condition][beta]) * 1.05)
            plt.xlim(10, 20)
        if c == 3:
            plt.xlabel('Trial', fontsize=20, position=(0.5, 0.5))
        elif c == 1:
            plt.ylabel('Shortcut Replays [#]', fontsize=20, position=(0, -0.5))
        ax = plt.gca()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
    art = []
    lgd = plt.legend(loc=9, bbox_to_anchor=(0.5, -0.5), ncol=5, fontsize=15, framealpha=0.)
    art.append(lgd)
    plt.savefig(file_name + '.png', dpi=200, bbox_inches='tight', transparent=True)
    plt.savefig(file_name + '.svg', dpi=200, bbox_inches='tight', transparent=True)
    plt.close('')
    
def plot_pooled(data, file_name):
    pooled = {}
    for beta in np.linspace(5, 15, 11):
        pooled[beta] = []
        for condition in data:
            pooled[beta] += [np.sum(data[condition][beta])]
    colors = plt.cm.jet(np.linspace(0, 1, 11))
    plt.figure(1)
    for b, beta in enumerate(pooled):
        plt.bar(np.arange(4) + 0.5/11 * b - 2.5/11, pooled[beta], 0.5/11, label='$\\beta_M = ' + str(beta) + '$', color=colors[b])
    plt.xticks(np.arange(4), np.array(list(data.keys())), fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylabel('Shortcuts [#]', fontsize=15)
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    art = []
    lgd = plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1), ncol=4, fontsize=12, framealpha=0.)
    art.append(lgd)
    plt.savefig(file_name + '.png', dpi=200, bbox_inches='tight', transparent=True)
    plt.savefig(file_name + '.svg', dpi=200, bbox_inches='tight', transparent=True)
    plt.close('all')
            
modes = ['default', 'reverse']
use_recency = [False, True]

center, sc, nsc = load_and_prepare('', modes, use_recency)
for mode in modes:
    for recency in use_recency:
        plot_details(sc[mode][recency], 'plots/mode_' + mode + '_recency_' + ('Y' if recency else 'N') + '_shortcuts_details')
        plot_pooled(sc[mode][recency], 'plots/mode_' + mode + '_recency_' + ('Y' if recency else 'N') + '_shortcuts_pooled')
