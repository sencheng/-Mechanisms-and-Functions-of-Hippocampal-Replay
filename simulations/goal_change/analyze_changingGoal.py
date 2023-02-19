# basic imports
import numpy as np
import pickle
import matplotlib.pyplot as plt


# params
modes = ['default', 'reverse', 'dynamic']
styles = {'default': '-', 'reverse': '-.', 'dynamic': '--'}

# load data
data = {'tmaze': {}, 'box': {}}
for mode in modes:
    data['tmaze'][mode] = pickle.load(open('data/changingGoal_tmaze_mode_' + mode + '.pkl', 'rb'))[mode]
    data['box'][mode] = pickle.load(open('data/changingGoal_box_mode_' + mode + '.pkl', 'rb'))[mode]
    
# plot data
plt.figure(1)
plt.suptitle('Learning Performance', position=(0.5, 1.03), fontsize=15)
plt.subplots_adjust(hspace=0.8)
plt.subplot(2, 1, 1)
plt.title('T-Maze')
for mode in modes:
    plt.plot(np.arange(600) + 1, np.mean(np.array(data['tmaze'][mode]), axis=0), linestyle=styles[mode])
plt.axvline(300, linewidth=1, linestyle='--', color='k')
plt.xlim(1, 600)
plt.ylim(13, 30)
plt.xlabel('Trial')
plt.ylabel('Escape Latency')
ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.subplot(2, 1, 2)
plt.title('Box')
for mode in modes:
    plt.plot(np.arange(600) + 1, np.mean(np.array(data['box'][mode]), axis=0), linestyle=styles[mode], label=mode)
plt.axvline(300, linewidth=1, linestyle='--', color='k')
plt.xlim(1, 600)
plt.ylim(13, 30)
plt.xlabel('Trial')
plt.ylabel('Escape Latency')
ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
art = []
lgd = plt.legend(loc=9, bbox_to_anchor=(0.5, -0.5), ncol=3, fontsize=12, framealpha=0.)
art.append(lgd)
plt.savefig('plots/changingGoal.png', dpi=200, bbox_inches='tight', transparent=True)
plt.savefig('plots/changingGoal.svg', dpi=200, bbox_inches='tight', transparent=True)
