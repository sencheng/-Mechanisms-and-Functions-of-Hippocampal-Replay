# basic imports
import numpy as np
import pickle
import matplotlib.pyplot as plt
            

def countInvalidTransitions(invalidTransitions, mode='default'):
    '''
    This function computes the fraction of invalid transitions
    that were replayed in the second and third environment.
    
    | **Args**
    | invalidTransitions:           Dictionary containing the invalid transition for environment 2 and environment 3.
    '''
    fractions = {'DR': {2: 0, 3: 0}, 'euclidean': {2: 0, 3: 0}}
    for d in ['DR', 'euclidean']:
        suffix = ''
        if d == 'DR':
            suffix = '_' + mode
        fileName = 'data/replays_' + d + suffix + '.pkl'
        replays = pickle.load(open(fileName, 'rb'))
        for i in range(2):
            invalid, E = 0, 0
            for replay in replays['env_' + str(i + 2)]:
                for e, experience in enumerate(replay):
                    t1, t2 = (experience['state'], replay[e - 1]['state']), (replay[e - 1]['state'], experience['state'])
                    if t1 in invalidTransitions[i + 2] or t2 in invalidTransitions[i + 2]:
                        invalid += 1
                    E += 1
            fractions[d][i + 2] = invalid/E
    
    return fractions

def plot(fractions, suffix='_default'):
    '''
    This function plots the fractions of invalid transitions for DR and Euclidean similarity metrics.
    
    | **Args**
    | fractions:                    Dictionary containing the fractions of invalid transitions.
    '''
    DR = np.array([fractions['DR'][2], fractions['DR'][3]]) * 100
    Euc = np.array([fractions['euclidean'][2], fractions['euclidean'][3]]) * 100
    w = 0.25
    plt.figure(1, figsize=(4, 4), facecolor='none')
    plt.bar(np.arange(2) - w/2, DR, width=w, color='b', edgecolor='k', linewidth=2, label='DR')
    plt.bar(np.arange(2) + w/2, Euc, width=w, color='r', edgecolor='k', linewidth=2, label='Euclidean')
    plt.xlim(-0.5, 1.5)
    #plt.ylim(0, 11)
    plt.xticks(np.arange(2), ['Environment 2', 'Environment 3'])
    plt.title('Invalid Transitions')
    plt.ylabel('Fraction [%]')
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    art = []
    lgd = plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1), ncol=5, fontsize=12, framealpha=0.)
    art.append(lgd)
    plt.savefig('plots/compare_dynamic' + suffix + '.png', dpi=200, bbox_inches='tight', transparent=True)
    plt.savefig('plots/compare_dynamic' + suffix + '.svg', dpi=200, bbox_inches='tight', transparent=True)
    plt.close('all')
                
if __name__ == "__main__":
    # invalid transitions of environment 2
    invalidTransitions_env_2 = [(1, 2), (11, 12), (21, 22), (31, 32), (41, 42), (51, 52), (61, 62), (71, 72)]
    invalidTransitions_env_2 += [(2, 1), (12, 11), (22, 21), (32, 31), (42, 41), (52, 51), (62, 61), (72, 71)]
    invalidTransitions_env_2 += [(14, 24), (15, 25), (16, 26), (17, 27)]
    invalidTransitions_env_2 += [(24, 14), (25, 15), (26, 16), (27, 17)]
    invalidTransitions_env_2 += [(23, 24), (33, 34), (43, 44), (53, 54), (63, 64), (73, 74), (83, 84), (93, 94)]
    invalidTransitions_env_2 += [(24, 23), (34, 33), (44, 43), (54, 53), (64, 63), (74, 73), (84, 83), (94, 93)]
    invalidTransitions_env_2 += [(36, 46), (37, 47), (38, 48), (39, 49)]
    invalidTransitions_env_2 += [(46, 36), (47, 37), (48, 38), (49, 39)]
    invalidTransitions_env_2 += [(76, 86), (77, 87), (86, 76), (87, 77)]
    invalidTransitions_env_2 += [(45, 46), (55, 56), (65, 66), (75, 76)]
    invalidTransitions_env_2 += [(46, 45), (56, 55), (66, 65), (76, 75)]
    # invalid transitions of environment 3
    invalidTransitions_env_3 = [(4, 5), (14, 15), (24, 25), (34, 35), (64, 65), (74, 75), (84, 85), (94, 95)]
    invalidTransitions_env_3 += [(5, 4), (15, 14), (25, 24), (35, 34), (65, 64), (75, 74), (85, 84), (95, 94)]
    invalidTransitions_env_3 += [(22, 23), (32, 33), (26, 27), (36, 37)]
    invalidTransitions_env_3 += [(23, 22), (33, 32), (27, 26), (37, 36)]
    invalidTransitions_env_3 += [(62, 63), (72, 73), (66, 67), (76, 77)]
    invalidTransitions_env_3 += [(63, 62), (73, 72), (67, 66), (77, 76)]
    invalidTransitions_env_3 += [(33, 43), (34, 44), (35, 45), (36, 46), (53, 63), (54, 64), (55, 65), (56, 66)]
    invalidTransitions_env_3 += [(43, 33), (44, 34), (45, 35), (46, 36), (63, 53), (64, 54), (65, 55), (66, 56)]
    # store invalid transition in dictionary
    invalidTransitions = {2: invalidTransitions_env_2, 3: invalidTransitions_env_3}
    
    # compute fractions of invalid transitions and plot them
    fractions = countInvalidTransitions(invalidTransitions)
    plot(fractions)
    fractions = countInvalidTransitions(invalidTransitions, 'reverse')
    plot(fractions, '_reverse')