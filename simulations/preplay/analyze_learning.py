# basic imports
import numpy as np
import pickle
import matplotlib.pyplot as plt


def compute_action_probabilities(q, beta):
    q = q[np.array([0, 2])]
    if np.sum(q == 0) == q.shape[0]:
        q.fill(1)
    p = np.exp(q * beta)
    p /= np.sum(p)
    
    return p

def plot(fileName, left, right):
    plt.figure(1, figsize=(4, 4))
    plt.bar(np.arange(2), left * 100, width=0.25, label='Uncued Arm')
    plt.bar(np.arange(2) + 0.25, right * 100, width=0.25, label='Cued Arm')
    plt.xticks(np.arange(2) + 0.125, ['Before', 'After'])
    plt.ylabel('Selection Probability [%]')
    plt.ylim(0, 100)
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    art = []
    lgd = plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=12, framealpha=0.)
    art.append(lgd)
    plt.savefig(fileName + '.png', dpi=200, bbox_inches='tight', transparent=True)
    plt.savefig(fileName + '.svg', dpi=200, bbox_inches='tight', transparent=True)
    plt.close('all')


if __name__ == "__main__":
    # params
    modes = ['default', 'reverse']
    gammas = [0.1, 0.8]
    beta = 20
    
    for gamma in gammas:
        for mode in modes:
            Qs = pickle.load(open('data/analyzed_learning_gamma_' + str(gamma) + '_mode_' + mode + '.pkl', 'rb'))
            before = Qs['before'][3]
            before = compute_action_probabilities(before, beta)
            after = Qs['after'][3]
            after = compute_action_probabilities(after, beta)
            plot('plots/learning_gamma_' + str(gamma) + '_mode_' + mode,
                 np.array([before[0], after[0]]), np.array([before[1], after[1]]))