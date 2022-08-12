# basic imports
import numpy as np
import pickle
import matplotlib.pyplot as plt


def plot(data, random, file_name):
    colors = plt.cm.jet(np.linspace(0, 1, 15))
    plt.figure(1)
    for b, beta in enumerate(data):
        plt.plot(np.arange(data[beta].shape[0]) + 1, data[beta], label='$\\beta_M = ' + str(beta) + '$', color=colors[b])
    plt.plot(np.arange(random.shape[0]) + 1, random, color='k')
    plt.xlabel('Trial')
    plt.ylabel('Escape Latency [#steps]')
    plt.xlim(1, data[beta].shape[0])
    plt.ylim(0, 55)
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    art = []
    lgd = plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1), ncol=5, fontsize=12, framealpha=0.)
    art.append(lgd)
    plt.savefig(file_name, dpi=200, bbox_inches='tight')
    plt.close('all')

# params
betas = np.linspace(1, 15, 15)
numbers_of_replays = [1, 10]
replay_types = ['local', 'step']
# load and average results
results, random = {}, {}
for replay_type in replay_types:
    results[replay_type] = {}
    random[replay_type] = {}
    for number_of_replays in numbers_of_replays:
        results[replay_type][number_of_replays] = {}
        random[replay_type][number_of_replays] = np.mean(np.array(pickle.load(open('data/random_type_' + replay_type + '_replays_' + str(number_of_replays) + '.pkl', 'rb'))), axis=0)
        for beta in betas:
            file_name = 'data/type_' + replay_type + '_replays_' + str(number_of_replays) + '_beta_' + str(beta) + '.pkl'
            results[replay_type][number_of_replays][beta] = np.mean(np.array(pickle.load(open(file_name, 'rb'))), axis=0)
# plot
plot(results['local'][10], random['local'][10], 'plots/learning_type_local_replays_10.png')
plot(results['local'][1], random['local'][1], 'plots/learning_type_local_replays_1.png')
plot(results['step'][10], random['step'][10], 'plots/learning_type_step_replays_10.png')
plot(results['step'][1], random['step'][1], 'plots/learning_type_step_replays_1.png')