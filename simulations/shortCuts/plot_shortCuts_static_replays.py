# basic imports
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt


def recover_states(replay):
    '''
    This function extracts the states of replayed experiences.
    
    | **Args**
    | replay:                       A sequence of replayed experiences.
    '''
    states = []
    for experience in replay:
        states += [experience['state']]
        
    return np.array(states)


def plot(replays, condition, beta):
    '''
    This function plots example replay sequences.
    
    | **Args**
    | replays:                      A list of replays (must be of length four or greater!).
    | condition:                    The simulation condition identifier.
    | beta:                         The inverse temperature parameter used by the replay mechanism.
    '''
    maps = np.zeros((4, 7, 11))
    for i in range(4):
        for s, state in enumerate(replays[i][:18]):
            x = int(state/11)
            y = state - x * 11
            maps[i, x, y] = s + 1
        maps[i] /= np.amax(maps[i])
            
    x1 = [0, 11, 11, 0, 0]
    y1 = [0, 0, 7, 7, 0]
    wall_1_x, wall_1_y = [1, 1], [1, 6]
    wall_2_x, wall_2_y = [10, 10], [1, 6]
    wall_3_x, wall_3_y = [5, 5], [1, 6]
    wall_4_x, wall_4_y = [6, 6], [1, 6]
    wall_5_x, wall_5_y = [1, 5], [1, 1]
    wall_6_x, wall_6_y = [6, 10], [1, 1]
    wall_7_x, wall_7_y = [1, 5], [6, 6]
    wall_8_x, wall_8_y = [6, 10], [6, 6]
    width = 3
    
    plt.figure(1, figsize=(18, 3))
    for i in range(4):
        plt.subplot(1, 4, i + 1)
        plt.pcolor(maps[i], cmap='hot')
        plt.plot(x1, y1, color='c', linewidth=width*2, zorder=100)
        plt.plot(wall_1_x, wall_1_y, color='c', linewidth=width, zorder=100)
        plt.plot(wall_2_x, wall_2_y, color='c', linewidth=width, zorder=100)
        plt.plot(wall_3_x, wall_3_y, color='c', linewidth=width, zorder=100)
        plt.plot(wall_4_x, wall_4_y, color='c', linewidth=width, zorder=100)
        plt.plot(wall_5_x, wall_5_y, color='c', linewidth=width, zorder=100)
        plt.plot(wall_6_x, wall_6_y, color='c', linewidth=width, zorder=100)
        plt.plot(wall_7_x, wall_7_y, color='c', linewidth=width, zorder=100)
        plt.plot(wall_8_x, wall_8_y, color='c', linewidth=width, zorder=100)
        #plt.xticks(np.array([0.5, 5.5, 10.5]), np.array([1, 6, 11]))
        #plt.yticks(np.array([0.5, 3.5, 6.5]), np.array([1, 4, 7]))
        plt.xticks([])
        plt.yticks([])
        
    plt.savefig('plots/replays/condition_' + condition + '_beta_' + str(beta) + '.png', dpi=200, bbox_inches='tight', transparent=True)
    plt.savefig('plots/replays/condition_' + condition + '_beta_' + str(beta) + '.svg', dpi=200, bbox_inches='tight', transparent=True)
    plt.close('all')


if __name__ == '__main__':
    # params
    betas = np.linspace(5, 15, 11)
    conditions = ['RL', 'AA', 'RA', 'AL']
    # load analysis information
    shortCuts_static_analyzed = pickle.load(open('analysis/shortCuts_static.pkl', 'rb'))
    
    # load and plot example short cut replays
    for condition in conditions:
        for beta in betas:
            # skip if there were no short cut replays
            idx = shortCuts_static_analyzed['default'][False][beta][condition]['idx_shortcuts']
            sc = shortCuts_static_analyzed['default'][False][beta][condition]['shortcuts']
            if np.sum(np.array(sc)) > 0.:
                # load replays
                file_name = 'data/static/mode_default_recency_N_beta_' + str(beta) + '.pkl'
                replays_raw = pickle.load(open(file_name, 'rb'))[condition]
                # gather replays from all trials
                replays = []
                for t, trial in enumerate(replays_raw):
                    for replay in idx[t]:
                        replays += [recover_states(trial[replay])]
                # plot replays
                if len(replays) >= 4:
                    plot(replays, condition, beta)
