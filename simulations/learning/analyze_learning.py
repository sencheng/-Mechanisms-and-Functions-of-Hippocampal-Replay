# basic imports
import numpy as np
import pickle
import matplotlib.pyplot as plt


def load_data():
    '''
    This function loads simulation data and prepares it.
    '''
    data = {}
    for env in ['linear_track', 'open_field', 'labyrinth']:
        data[env] = {}
        # load raw data
        data[env]['reverse'] = pickle.load(open('data/' + env + '_SFMA_reverse.pkl', 'rb'))
        data[env]['default'] = pickle.load(open('data/' + env + '_SFMA_default.pkl', 'rb'))
        data[env]['dynamic'] = pickle.load(open('data/' + env + '_SFMA_dynamic.pkl', 'rb'))
        data[env]['online'] = pickle.load(open('data/' + env + '_online.pkl', 'rb'))
        data[env]['random'] = pickle.load(open('data/' + env + '_random.pkl', 'rb'))
        data[env]['pma'] = pickle.load(open('data/' + env + '_PMA.pkl', 'rb'))
        # compute averages for reverse and default modes across all discount factors
        for mode in ['reverse', 'default', 'dynamic']:
            data[env][mode + '_avg'] = []
            for gamma in data[env][mode]:
                data[env][mode + '_avg'] += data[env][mode][gamma]
            data[env][mode + '_avg'] = np.mean(np.array(data[env][mode + '_avg']), axis=0)
            
    return data

def plot_separate(linear_track, open_field, labyrinth, suffix=''):
    '''
    This function plots the learning performance of RL agents using our replay method.
    Performance is plotted separately for different Default Representation discount factors.
    
    | **Args**
    | linear_track:                 Data collected in linear track environment for our model.
    | open_field:                   Data collected in open field environment for our model.
    | labyrinth:                    Data collected in labyrinth environment for our model.
    | suffix:                       Optional suffix that will be applied to the file name.
    '''
    plt.figure(1, figsize=(5, 5))
    plt.subplots_adjust(hspace=0.7)
    plt.subplot(3, 1, 1)
    for gamma in linear_track:
        plt.plot(np.arange(20) + 1, np.mean(np.array(linear_track[gamma]), axis=0), label=gamma)
    plt.axhline(9, linestyle='--', color='grey')
    plt.xticks(np.array([1, 5, 10, 15, 20]), np.array([1, 5, 10, 15, 20]))
    plt.ylim(0, None)
    plt.xlim(1, 20)
    plt.title('Linear Track')
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.subplot(3, 1, 2)
    for gamma in open_field:
        plt.plot(np.arange(100) + 1, np.mean(np.array(open_field[gamma]), axis=0), label=gamma)
    plt.axhline(10, linestyle='--', color='grey')
    plt.xticks(np.array([1, 20, 40, 60, 80, 100]), np.array([1, 20, 40, 60, 80, 100]))
    plt.ylim(0, None)
    plt.xlim(1, 100)
    plt.title('Open Field')
    plt.ylabel('Escape Latency [#steps]')
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.subplot(3, 1, 3)
    for gamma in labyrinth:
        plt.plot(np.arange(100) + 1, np.mean(np.array(labyrinth[gamma]), axis=0), label=gamma)
    plt.axhline(18, linestyle='--', color='grey')
    plt.xticks(np.array([1, 20, 40, 60, 80, 100]), np.array([1, 20, 40, 60, 80, 100]))
    plt.ylim(0, None)
    plt.xlim(1, 100)
    plt.title('Labyrinth')
    plt.xlabel('Trial')
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    art = []
    lgd = plt.legend(loc=9, bbox_to_anchor=(0.5, -0.55), ncol=5, fontsize=10)
    art.append(lgd)
    plt.savefig('plots/learning_separate' + suffix + '.png', dpi=200, bbox_inches='tight')
    plt.savefig('plots/learning_separate' + suffix + '.svg', dpi=200, bbox_inches='tight')
    plt.close('all')

def plot_learning(linear_track, open_field, labyrinth, suffix=''):
    '''
    This function plots the learning performance of RL agents using our replay method
    as well as online, randam and PMA agents for comparison.
    
    | **Args**
    | linear_track:                 Data collected in linear track environment for all agent types.
    | open_field:                   Data collected in open field environment for all agent types.
    | labyrinth:                    Data collected in labyrinth environment for all agent types.
    | suffix:                       Optional suffix that will be applied to the file name.
    '''
    plt.figure(1, figsize=(5, 5))
    plt.subplots_adjust(hspace=0.7)
    plt.subplot(3, 1, 1)
    plt.plot(np.arange(20) + 1, linear_track['sfma'], color='g', label='SFMA')
    plt.plot(np.arange(20) + 1, np.mean(np.array(linear_track['random']), axis=0), color='b', label='Random')
    plt.plot(np.arange(20) + 1, np.mean(np.array(linear_track['online']), axis=0), color='k', label='Online')
    plt.plot(np.arange(20) + 1, np.mean(np.array(linear_track['pma']), axis=0), color='r', label='PMA')
    plt.axhline(9, linestyle='--', color='grey')
    plt.xticks(np.array([1, 5, 10, 15, 20]), np.array([1, 5, 10, 15, 20]))
    plt.ylim(0, None)
    plt.xlim(1, 20)
    plt.title('Linear Track')
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.subplot(3, 1, 2)
    plt.plot(np.arange(100) + 1, open_field['sfma'], color='g', label='SFMA')
    plt.plot(np.arange(100) + 1, np.mean(np.array(open_field['random']), axis=0), color='b', label='Random')
    plt.plot(np.arange(100) + 1, np.mean(np.array(open_field['online']), axis=0), color='k', label='Online')
    plt.plot(np.arange(100) + 1, np.mean(np.array(open_field['pma']), axis=0), color='r', label='PMA')
    plt.axhline(10, linestyle='--', color='grey')
    plt.xticks(np.array([1, 20, 40, 60, 80, 100]), np.array([1, 20, 40, 60, 80, 100]))
    plt.ylim(0, None)
    plt.xlim(1, 100)
    plt.title('Open Field')
    plt.ylabel('Escape Latency [#steps]')
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.subplot(3, 1, 3)
    plt.plot(np.arange(100) + 1, labyrinth['sfma'], color='g', label='SFMA')
    plt.plot(np.arange(100) + 1, np.mean(np.array(labyrinth['random']), axis=0), color='b', label='Random')
    plt.plot(np.arange(100) + 1, np.mean(np.array(labyrinth['online']), axis=0), color='k', label='Online')
    plt.plot(np.arange(100) + 1, np.mean(np.array(labyrinth['pma']), axis=0), color='r', label='PMA')
    plt.axhline(18, linestyle='--', color='grey')
    plt.xticks(np.array([1, 20, 40, 60, 80, 100]), np.array([1, 20, 40, 60, 80, 100]))
    plt.ylim(0, None)
    plt.xlim(1, 100)
    plt.title('Labyrinth')
    plt.xlabel('Trial')
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    art = []
    lgd = plt.legend(loc=9, bbox_to_anchor=(0.5, -0.55), ncol=3, fontsize=10)
    art.append(lgd)
    plt.savefig('plots/learning' + suffix + '.png', dpi=200, bbox_inches='tight')
    plt.savefig('plots/learning' + suffix + '.svg', dpi=200, bbox_inches='tight')
    plt.close('all')
    
def plot_learning_final(linear_track, open_field, labyrinth, suffix=''):
    '''
    This function plots the learning performance of RL agents using our replay method
    as well as online, randam and PMA agents for comparison.
    
    | **Args**
    | linear_track:                 Data collected in linear track environment for all agent types.
    | open_field:                   Data collected in open field environment for all agent types.
    | labyrinth:                    Data collected in labyrinth environment for all agent types.
    | suffix:                       Optional suffix that will be applied to the file name.
    '''
    plt.figure(1, figsize=(5, 5))
    plt.subplots_adjust(hspace=0.7)
    plt.subplot(3, 1, 1)
    plt.plot(np.arange(20) + 1, linear_track['sfma_def'], color='g', linestyle='-', label='SFMA (Default)')
    plt.plot(np.arange(20) + 1, linear_track['sfma_rev'], color='g', linestyle='-.', label='SFMA (Reverse)')
    plt.plot(np.arange(20) + 1, linear_track['sfma_dyn'], color='g', linestyle='--', label='SFMA (Dynamic)')
    plt.plot(np.arange(20) + 1, np.mean(np.array(linear_track['random']), axis=0), color='b', label='Random')
    plt.plot(np.arange(20) + 1, np.mean(np.array(linear_track['online']), axis=0), color='k', label='Online')
    plt.plot(np.arange(20) + 1, np.mean(np.array(linear_track['pma']), axis=0), color='r', label='PMA')
    plt.axhline(9, linestyle='--', color='grey')
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
    plt.plot(np.arange(100) + 1, np.mean(np.array(open_field['random']), axis=0), color='b', label='Random')
    plt.plot(np.arange(100) + 1, np.mean(np.array(open_field['online']), axis=0), color='k', label='Online')
    plt.plot(np.arange(100) + 1, np.mean(np.array(open_field['pma']), axis=0), color='r', label='PMA')
    plt.axhline(10, linestyle='--', color='grey')
    plt.xticks(np.array([1, 20, 40, 60, 80, 100]), np.array([1, 20, 40, 60, 80, 100]))
    plt.ylim(0, None)
    plt.xlim(1, 100)
    plt.title('Open Field')
    plt.ylabel('Escape Latency [#steps]')
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.subplot(3, 1, 3)
    plt.plot(np.arange(100) + 1, labyrinth['sfma_def'], color='g', linestyle='-', label='SFMA (Default)')
    plt.plot(np.arange(100) + 1, labyrinth['sfma_rev'], color='g', linestyle='-.', label='SFMA (Reverse)')
    plt.plot(np.arange(100) + 1, labyrinth['sfma_dyn'], color='g', linestyle='--', label='SFMA (Dynamic)')
    plt.plot(np.arange(100) + 1, np.mean(np.array(labyrinth['random']), axis=0), color='b', label='Random')
    plt.plot(np.arange(100) + 1, np.mean(np.array(labyrinth['online']), axis=0), color='k', label='Online')
    plt.plot(np.arange(100) + 1, np.mean(np.array(labyrinth['pma']), axis=0), color='r', label='PMA')
    plt.axhline(18, linestyle='--', color='grey')
    plt.xticks(np.array([1, 20, 40, 60, 80, 100]), np.array([1, 20, 40, 60, 80, 100]))
    plt.ylim(0, None)
    plt.xlim(1, 100)
    plt.title('Labyrinth')
    plt.xlabel('Trial')
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    art = []
    lgd = plt.legend(loc=9, bbox_to_anchor=(0.5, -0.55), ncol=3, fontsize=10, framealpha=0.)
    art.append(lgd)
    plt.savefig('plots/learning' + suffix + '_final.png', dpi=200, bbox_inches='tight', transparent=True)
    plt.savefig('plots/learning' + suffix + '_final.svg', dpi=200, bbox_inches='tight', transparent=True)
    plt.close('all')


# load data
data = load_data()

# plot data
plot_separate(data['linear_track']['default'], data['open_field']['default'], data['labyrinth']['default'], suffix='_default')
plot_separate(data['linear_track']['reverse'], data['open_field']['reverse'], data['labyrinth']['reverse'], suffix='_reverse')
plot_separate(data['linear_track']['dynamic'], data['open_field']['dynamic'], data['labyrinth']['dynamic'], suffix='_dynamic')
plot_learning({'sfma': data['linear_track']['default_avg'], 'online': data['linear_track']['online'], 'random': data['linear_track']['random'], 'pma': data['linear_track']['pma']},
              {'sfma': data['open_field']['default_avg'], 'online': data['open_field']['online'], 'random': data['open_field']['random'], 'pma': data['open_field']['pma']},
              {'sfma': data['labyrinth']['default_avg'], 'online': data['labyrinth']['online'], 'random': data['labyrinth']['random'], 'pma': data['labyrinth']['pma']},
              suffix='_default')
plot_learning({'sfma': data['linear_track']['reverse_avg'], 'online': data['linear_track']['online'], 'random': data['linear_track']['random'], 'pma': data['linear_track']['pma']},
              {'sfma': data['open_field']['reverse_avg'], 'online': data['open_field']['online'], 'random': data['open_field']['random'], 'pma': data['open_field']['pma']},
              {'sfma': data['labyrinth']['reverse_avg'], 'online': data['labyrinth']['online'], 'random': data['labyrinth']['random'], 'pma': data['labyrinth']['pma']},
              suffix='_reverse')
plot_learning({'sfma': data['linear_track']['dynamic_avg'], 'online': data['linear_track']['online'], 'random': data['linear_track']['random'], 'pma': data['linear_track']['pma']},
              {'sfma': data['open_field']['dynamic_avg'], 'online': data['open_field']['online'], 'random': data['open_field']['random'], 'pma': data['open_field']['pma']},
              {'sfma': data['labyrinth']['dynamic_avg'], 'online': data['labyrinth']['online'], 'random': data['labyrinth']['random'], 'pma': data['labyrinth']['pma']},
              suffix='_dynamic')
plot_learning_final({'sfma_def': data['linear_track']['default_avg'], 'sfma_rev': data['linear_track']['reverse_avg'], 'sfma_dyn': data['linear_track']['dynamic_avg'], 'online': data['linear_track']['online'], 'random': data['linear_track']['random'], 'pma': data['linear_track']['pma']},
              {'sfma_def': data['open_field']['default_avg'], 'sfma_rev': data['open_field']['reverse_avg'], 'sfma_dyn': data['open_field']['dynamic_avg'], 'online': data['open_field']['online'], 'random': data['open_field']['random'], 'pma': data['open_field']['pma']},
              {'sfma_def': data['labyrinth']['default_avg'], 'sfma_rev': data['labyrinth']['reverse_avg'], 'sfma_dyn': data['labyrinth']['dynamic_avg'], 'online': data['labyrinth']['online'], 'random': data['labyrinth']['random'], 'pma': data['labyrinth']['pma']},
              suffix='')
