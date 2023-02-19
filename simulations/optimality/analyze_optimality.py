# basic imports
import numpy as np
import pickle
import matplotlib.pyplot as plt


def load_data() -> dict:
    '''
    This function loads simulation data and prepares it.
    
    Parameters
    ----------
    None
    
    Returns
    ----------
    data :                              The prepared data.
    '''
    data = {}
    for env in ['linear_track', 'open_field', 'labyrinth']:
        data[env] = {}
        # load raw data
        data[env]['reverse'] = pickle.load(open('data/optimality_' + env + '_SFMA_reverse.pkl', 'rb'))
        data[env]['default'] = pickle.load(open('data/optimality_' + env + '_SFMA_default.pkl', 'rb'))
        data[env]['dynamic'] = pickle.load(open('data/optimality_' + env + '_SFMA_dynamic.pkl', 'rb'))
        # compute averages for reverse and default modes across all discount factors
        for mode in ['reverse', 'default', 'dynamic']:
            data[env][mode + '_avg'] = np.mean(data[env][mode], axis=0)
            
    return data
    
def plot_optimality(linear_track: dict, open_field: dict, labyrinth: dict):
    '''
    This function plots the learning performance of RL agents using our replay method
    as well as online, random and PMA agents for comparison.
    
    Parameters
    ----------
    linear_track :                      Data collected in linear track environment for all agent types.
    open_field :                        Data collected in open field environment for all agent types.
    labyrinth :                         Data collected in labyrinth environment for all agent types.
    
    Returns
    ----------
    None
    '''
    plt.figure(1, figsize=(5, 5))
    plt.subplots_adjust(hspace=0.7)
    plt.subplot(3, 1, 1)
    plt.plot(np.arange(20) + 1, linear_track['sfma_def'], color='g', linestyle='-', label='SFMA (Default)')
    plt.plot(np.arange(20) + 1, linear_track['sfma_rev'], color='g', linestyle='-.', label='SFMA (Reverse)')
    plt.plot(np.arange(20) + 1, linear_track['sfma_dyn'], color='g', linestyle='--', label='SFMA (Dynamic)')
    plt.xticks(np.array([1, 5, 10, 15, 20]), np.array([1, 5, 10, 15, 20]))
    plt.ylim(0, None)
    plt.xlim(1, 20)
    plt.title('Linear Track')
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.subplot(3, 1, 2)
    plt.plot(np.arange(100) + 1, open_field['sfma_def'], color='g', linestyle='-', label='SFMA (Default)')
    plt.plot(np.arange(100) + 1, open_field['sfma_rev'], color='g', linestyle='-.', label='SFMA (Reverse)')
    plt.plot(np.arange(100) + 1, open_field['sfma_dyn'], color='g', linestyle='--', label='SFMA (Dynamic)')
    plt.xticks(np.array([1, 20, 40, 60, 80, 100]), np.array([1, 20, 40, 60, 80, 100]))
    plt.ylim(0, None)
    plt.xlim(1, 20)
    plt.title('Open Field')
    plt.ylabel('Optimal Updates [#]')
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.subplot(3, 1, 3)
    plt.plot(np.arange(100) + 1, labyrinth['sfma_def'], color='g', linestyle='-', label='SFMA (Default)')
    plt.plot(np.arange(100) + 1, labyrinth['sfma_rev'], color='g', linestyle='-.', label='SFMA (Reverse)')
    plt.plot(np.arange(100) + 1, labyrinth['sfma_dyn'], color='g', linestyle='--', label='SFMA (Dynamic)')
    plt.xticks(np.array([1, 20, 40, 60, 80, 100]), np.array([1, 20, 40, 60, 80, 100]))
    plt.ylim(0, None)
    plt.xlim(1, 20)
    plt.title('Labyrinth')
    plt.xlabel('Trial')
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    art = []
    lgd = plt.legend(loc=9, bbox_to_anchor=(0.5, -0.55), ncol=3, fontsize=10, framealpha=0.)
    art.append(lgd)
    plt.savefig('plots/optimality.png', dpi=200, bbox_inches='tight', transparent=True)
    plt.savefig('plots/optimality.svg', dpi=200, bbox_inches='tight', transparent=True)
    plt.close('all')


# load data
data = load_data()

# plot data
plot_optimality({'sfma_def': data['linear_track']['default_avg'], 'sfma_rev': data['linear_track']['reverse_avg'], 'sfma_dyn': data['linear_track']['dynamic_avg']},
              {'sfma_def': data['open_field']['default_avg'], 'sfma_rev': data['open_field']['reverse_avg'], 'sfma_dyn': data['open_field']['dynamic_avg']},
              {'sfma_def': data['labyrinth']['default_avg'], 'sfma_rev': data['labyrinth']['reverse_avg'], 'sfma_dyn': data['labyrinth']['dynamic_avg']})
