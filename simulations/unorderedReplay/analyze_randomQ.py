# basic imports
import numpy as np
import pickle
import matplotlib.pyplot as plt


# paramas
modes = ['default', 'reverse', 'dynamic']

# load data
data = {'tmaze': {}, 'box': {}}
for mode in modes:
    data['tmaze'][mode] = pickle.load(open('data/randomQ_tmaze_mode_' + mode + '.pkl', 'rb'))[mode]
    data['box'][mode] = pickle.load(open('data/randomQ_box_mode_' + mode + '.pkl', 'rb'))[mode]
    
# plot data
plt.figure(1)
plt.suptitle('Learning Performance', position=(0.5, 1.03), fontsize=15)
plt.subplots_adjust(hspace=0.8)
plt.subplot(2, 1, 1)
plt.title('T-Maze')
for mode in modes:
    plt.plot(np.arange(250) + 1, np.mean(np.array(data['tmaze'][mode]), axis=0))
    plt.xlim(1, 250)
    plt.ylim(13, 30)
    plt.xlabel('Trial')
    plt.ylabel('Escape Latency')
plt.subplot(2, 1, 2)
plt.title('Box')
for mode in modes:
    plt.plot(np.arange(250) + 1, np.mean(np.array(data['box'][mode]), axis=0), label=mode)
    plt.xlim(1, 250)
    plt.ylim(13, 30)
    plt.xlabel('Trial')
    plt.ylabel('Escape Latency')
art = []
lgd = plt.legend(loc=9, bbox_to_anchor=(0.5, -0.5), ncol=3, fontsize=12)
art.append(lgd)
plt.savefig('plots/randomQ.png', dpi=200, bbox_inches='tight')