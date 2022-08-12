# basic imports
import numpy as np
import pickle
import matplotlib.pyplot as plt
# sklearn for linear regression
from sklearn.linear_model import LinearRegression


def retrieve_replays(gammas, decays, height, width, occupancy, mode):
    '''
    This function retrieves the replayed positions of the replays
    produced by the SMA agent's memory module for different strength decay factors.
    
    | **Args**
    | gammas:                       The discount factors use for the DR.
    | decays:                       The decay factors.
    | height:                       The height of the gridworld environment.
    | width:                        The width of the gridworld environment.
    | occupancy:                    Predefined occupancy.
    | mode:                         The replay mode that was used.
    '''
    results = dict()
    for gamma in gammas:
        results[gamma] = dict()
        for decay in decays:
            results[gamma][decay] = []
            file_name = 'data/BDA/occupancy_' + occupancy + '_mode_' + mode + '_gamma_' + str(gamma) + '_decay_' + str(decay) + '.pkl'
            replays = pickle.load(open(file_name, 'rb'))
            for replay in replays:
                results[gamma][decay] += [[]]
                for experience in replay:
                    state = experience['state']
                    x = int(state/100)
                    y = state - x * 100
                    results[gamma][decay][-1] += [[x, y]]
                results[gamma][decay][-1] = np.array(results[gamma][decay][-1])
            
    return results

def trajectory_analysis(replays, intervals=[1, 2], regression_intervals=None):
    '''
    This function performs linear regression between the time intervals (in steps) and
    the average distances (in states) between replayed states.
    
    | **Args**
    | replays:                      The replays produced by the SMA agent's memory module.
    | intervals:                    The intervals for which the average distances are computed.
    | regression_intervals:         The intervals for which the linear regression is performed. If 'None', intervals is used.
    '''
    results = dict()
    for interval in intervals:
        results[interval] = []
        for replay in replays:
            for step in range(0, len(replay)-interval, interval):
                results[interval] += [ np.sqrt(np.sum((replay[step]-replay[step+interval])**2)) ]
        results[interval] = np.mean(np.array(results[interval]))
    
    if not regression_intervals is None:
        intervals = regression_intervals
    
    log_intervals = np.log10(np.array([[interval] for interval in intervals]))
    log_distances = np.log10(np.array([results[interval] for interval in intervals]))
    reg = LinearRegression().fit(log_intervals, log_distances)
    score = reg.score(log_intervals, log_distances)
        
    return {'distance': results, 'alpha': reg.coef_[0], 'G': np.power(10, reg.intercept_), 'score': score}

def plot_relationship(sweep, suffix=''):
    '''
    This function plots the relationship between time-step interval and average distance of reactivated experiences on a log-log scale.
    
    | **Args**
    | sweep:                        The replays produced by the SMA agent's memory module.
    | suffix:                       Optional suffix that will be add to the file name.
    '''
    # plot time-step intervals vs average distance on log scale for parameter combinations
    plt.figure(1)
    plt.title('Parameter Sweep\nDiscount Factor ($\gamma_{DR}$) @ $\lambda_{I}=0.9$', position=(0.5, 1.05), fontsize=20)
    plt.xlabel('Time-step Interval ($\Delta$t)', fontsize=15)
    plt.ylabel('Average Distance of\nReplayed Experiences (#states)', fontsize=15)
    plt.xscale('log')
    plt.yscale('log')
    for gamma in sweep:
        plt.plot(intervals, np.array(list(sweep[gamma][0.9]['distance'].values())), linewidth=1, label=str(gamma))
    art = []
    lgd = plt.legend(loc=9, bbox_to_anchor=(0.5, -0.2), ncol=5, framealpha=0., fontsize=10)
    art.append(lgd)
    plt.savefig('plots/randomWalk_relationship' + suffix + '.png', dpi=300, bbox_inches='tight', facecolor='none', edgecolor='none')
    plt.savefig('plots/randomWalk_relationship' + suffix + '.svg', dpi=300, bbox_inches='tight', facecolor='none', edgecolor='none')
    plt.close('all')
    
def plot_sweep(heatmaps, suffix=''):
    '''
    This function plots the regression results for different values of DR discount factor and inhibition decay as heatmaps.
    
    | **Args**
    | heatmaps:                     The replays produced by the SMA agent's memory module.
    | suffix:                       Optional suffix that will be add to the file name.
    '''
    # plot heatmaps for alpha and G
    plt.figure(2, figsize=(9.5, 3.5))
    plt.subplots_adjust(wspace=0.4)
    plt.suptitle('Regression Coefficients and Intercepts\nEqual Experience Strength', position=(0.5, 1.15), fontsize=20)
    plt.subplot(1, 2, 1)
    plt.title('Anomaly Parameter ($\\alpha$)', fontsize=18, position=(0.5, 1.05))
    plt.pcolor(heatmaps['alpha'], cmap='hot', vmin=0.30, vmax=0.70)
    plt.xlabel('Inhibition Decay ($\lambda_{I}$)', fontsize=15)
    plt.xticks(np.arange(0.5, 10.5), ['0.9', '0.8', '0.7', '0.6', '0.5', '0.4', '0.3', '0.2', '0.1', '0.0'])
    plt.ylabel('Discount Factor ($\gamma_{DR}$)', fontsize=15)
    plt.yticks(np.arange(0.5, 10.5), ['0.01', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9'])
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=15)
    plt.subplot(1, 2, 2)
    plt.title('Diffusion Coefficient (G)', fontsize=18, position=(0.5, 1.05))
    plt.pcolor(heatmaps['G'], cmap='hot', vmin=0.5, vmax=1.5)
    plt.xlabel('Inhibition Decay ($\lambda_{I}$)', fontsize=15)
    plt.xticks(np.arange(0.5, 10.5), ['0.9', '0.8', '0.7', '0.6', '0.5', '0.4', '0.3', '0.2', '0.1', '0.0'])
    plt.ylabel('Discount Factor ($\gamma_{DR}$)', fontsize=15)
    plt.yticks(np.arange(0.5, 10.5), ['0.01', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9'])
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=15)
    plt.savefig('plots/randomWalk_sweep_RegressionResults' + suffix + '.png', dpi=300, bbox_inches='tight', transparent=True)
    plt.savefig('plots/randomWalk_sweep_RegressionResults' + suffix + '.svg', dpi=300, bbox_inches='tight', transparent=True)
    # plot heatmaps for alpha and G separately
    plt.figure(3, figsize=(4, 3.5))
    plt.title('Anomaly Parameter ($\\alpha$)', fontsize=20, position=(0.5, 1.05))
    plt.pcolor(heatmaps['alpha'], cmap='hot', vmin=0.30, vmax=0.70)
    plt.xlabel('Inhibition Decay ($\lambda_{I}$)', fontsize=18)
    plt.xticks(np.arange(0.5, 10.5), ['0.9', '0.8', '0.7', '0.6', '0.5', '0.4', '0.3', '0.2', '0.1', '0.0'])
    plt.ylabel('Discount Factor ($\gamma_{DR}$)', fontsize=18)
    plt.yticks(np.arange(0.5, 10.5), ['0.01', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9'])
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=15)
    plt.savefig('plots/randomWalk_sweep_alpha' + suffix + '.png', dpi=300, bbox_inches='tight', transparent=True)
    plt.savefig('plots/randomWalk_sweep_alpha' + suffix + '.svg', dpi=300, bbox_inches='tight', transparent=True)
    plt.figure(4, figsize=(4, 3.5))
    plt.title('Diffusion Coefficient (G)', fontsize=20, position=(0.5, 1.05))
    plt.pcolor(heatmaps['G'], cmap='hot', vmin=0.5, vmax=1.5)
    plt.xlabel('Inhibition Decay ($\lambda_{I}$)', fontsize=18)
    plt.xticks(np.arange(0.5, 10.5), ['0.9', '0.8', '0.7', '0.6', '0.5', '0.4', '0.3', '0.2', '0.1', '0.0'])
    plt.ylabel('Discount Factor ($\gamma_{DR}$)', fontsize=18)
    plt.yticks(np.arange(0.5, 10.5), ['0.01', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9'])
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=15)
    plt.savefig('plots/randomWalk_sweep_G' + suffix + '.png', dpi=300, bbox_inches='tight', transparent=True)
    plt.savefig('plots/randomWalk_sweep_G' + suffix + '.svg', dpi=300, bbox_inches='tight', transparent=True)
    plt.close('all')
    
def plot_replays(replays, map_size=21, suffix=''):
    '''
    This function plots the replays as a heatmap of reactivated states.
    Reactivated positions are plotted relative to the starting position.
    
    | **Args**
    | replays:                      The list of replays.
    | map_size:                     The size of the heatmap.
    | suffix:                       A suffix that will be appended to the file name.
    '''
    plt.figure(1, figsize=(5, 5))
    for i in range(min(9, len(replays))):
        plt.subplot(3, 3, i + 1)
        replay_map = np.zeros((map_size, map_size))
        start = np.array(replays[i][0])
        for p, pos in enumerate(replays[i][:25]):
            x, y = pos - start
            if x + 10 >= 0 and x + 10 < map_size and y + 10 >= 0 and y + 10 < map_size:
                replay_map[x + 10, y + 10] = p
        replay_map /= p
        plt.pcolor(replay_map, cmap='hot', vmin=0, vmax=1)
        plt.xticks([])
        plt.yticks([])
    plt.savefig('plots/replays' + suffix + '.png', dpi=300, bbox_inches='tight', facecolor='none', edgecolor='none')
    plt.savefig('plots/replays' + suffix + '.svg', dpi=300, bbox_inches='tight', facecolor='none', edgecolor='none')
    plt.close('all')


if __name__ == "__main__":
    # define parameters
    height, width = 100, 100 # environment dimensions
    gammas = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] # discount factors
    lambdas = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0] # inhibition decay factors
    intervals = [1, 2, 5, 10, 25, 50, 100, 200, 400] # regression intervals
    occupancies = ['uniform', 'heterogeneous'] # predefined occupancies
    modes = ['default', 'reverse'] # replay modes
    
    # analyze replays
    for occupancy in occupancies:
        for mode in modes:
            print('Occupancy: ' + occupancy + ', Mode: ' + mode)
            replays = retrieve_replays(gammas, lambdas, height, width, occupancy, mode)
            # perform BDA analysis for the different values of gamma and lambda
            sweep = {}
            for gamma in gammas:
                sweep[gamma] = {}
                for decay in lambdas:
                    sweep[gamma][decay] = trajectory_analysis(replays[gamma][decay], intervals)
            # plot time-step intervals vs average distance on log scale for parameter combinations
            plot_relationship(sweep, '_' + occupancy + '_' + mode)
            # prepare heatmaps
            heatmaps = {'alpha': np.zeros((len(gammas), len(lambdas))), 'G': np.zeros((len(gammas), len(lambdas)))}
            for g, gamma in enumerate(gammas):
                for d, decay in enumerate(lambdas):
                    heatmaps['alpha'][g, d] = sweep[gamma][decay]['alpha']
                    heatmaps['G'][g, d] = sweep[gamma][decay]['G']
                    if gamma == 0.1 and decay == 0.9:
                        print(gamma, decay, sweep[gamma][decay]['alpha'])
                    if gamma == 0.8 and decay == 0.9:
                        print(gamma, decay, sweep[gamma][decay]['alpha'])
            # plot heatmaps
            plot_sweep(heatmaps, '_' + occupancy + '_' + mode)
            
    # load and plot replays
    replays = retrieve_replays(gammas, lambdas, height, width, 'uniform', 'default')
    for gamma in [0.1, 0.5, 0.9]:
        for decay in [0.0, 0.5, 0.9]:
            plot_replays(replays[0.1][0.9], suffix='_gamma_' + str(gamma) + '_decay_' + str(decay))