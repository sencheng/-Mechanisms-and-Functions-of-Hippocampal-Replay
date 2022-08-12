# basic imports
import numpy as np
import pickle
import matplotlib.pyplot as plt


def count_points(replays, width, height, bin_size=1):
    '''
    This function counts how often replays start in a given area of the environment.
    
    | **Args**
    | replays:                      A list of generated replays.
    | width:                        The width of the environment in number of states.
    | height:                       The height of the environment in number of states.
    | bin_size:                     The bin size.
    '''
    R = np.zeros((int(height/bin_size), int(width/bin_size)))
    for replay in replays:
        x, y = replay[0]
        x = int(x/bin_size)
        y = int(y/bin_size)
        R[x, y] += 1
        
    return R

def compute_randomness(R, width, height):
    '''
    This function computes the randomness of replay start positions.
    
    | **Args**
    | R:                            An array countaining the amount of replay starts for each area.
    | width:                        The width of the environment in number of states.
    | height:                       The height of the environment in number of states.
    '''
    P = R/np.sum(R)
    H = - np.sum(P * np.log(P))
    H_max = np.log(width * height)
    
    return H, H_max

def compute_directions(replays, width, height, bin_size=20, dir_bin_size=90):
    '''
    This function counts how often replays moves into a certain orientation for different areas of the environment.
    
    | **Args**
    | replays:                      A list of generated replays.
    | width:                        The width of the environment in number of states.
    | height:                       The height of the environment in number of states.
    | bin_size:                     The spatial bin size.
    | dir_bin_size:                 The  directional bin size.
    '''
    D = np.zeros((int(height/bin_size), int(width/bin_size), int(360/dir_bin_size)))
    for replay in replays:
        p1 = np.array(replay[0])
        p2 = np.array(replay[1])
        p2 -= p1
        d = np.rad2deg(np.arctan2(p2[0], p2[1])) + dir_bin_size/2
        d %= 360
        x = int(p1[0]/bin_size)
        y = int(p1[1]/bin_size)
        a = int(d/dir_bin_size)
        D[x, y, a] += 1
        
    return D

def compute_reactivations(replays, width, height, view_width, view_height):
    '''
    This function computes the diffusion of replay for a couple of time steps.
    
    | **Args**
    | replays:                      A list of generated replays.
    | width:                        The width of the environment in number of states.
    | height:                       The height of the environment in number of states.
    | view_width:                   The width of the view field.
    | view_height:                  The height of the view field.
    '''
    R = np.zeros((view_height, view_width, 4))
    center = np.array([int(view_height/2), int(view_width/2)])
    for replay in replays:
        start = np.array(replay[0])
        for i in range(4):
            offset = center + (np.array(replay[i + 1]) - start)
            try:
                R[offset[0], offset[1], i] += 1
            except:
                pass
            
    return R

def plot_starting_distribution(R, bin_size=1, suffix=''):
    '''
    This function plots the distribution of positions were replay started.
    
    | **Args**
    | R:                            The distribution of starting positions.
    | bin_size:                     The bin size.
    '''
    Dist = np.zeros((int(R.shape[0]/bin_size), int(R.shape[1]/bin_size)))
    for x in range(R.shape[0]):
        for y in range(R.shape[1]):
            x_new, y_new = int(x/bin_size), int(y/bin_size)
            Dist[x_new, y_new] = R[x, y]
            
    Dist /= np.sum(Dist)
    plt.figure(1, figsize=(6, 5))
    plt.pcolor(np.flip(Dist, axis=0), cmap='hot')
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('X Position', fontsize=15)
    plt.ylabel('Y Position', fontsize=15)
    plt.colorbar()
    plt.savefig('plots/distribution_starting_positions' + suffix + '.png', dpi=200, bbox_inches='tight', transparent=True)
    plt.savefig('plots/distribution_starting_positions' + suffix + '.svg', dpi=200, bbox_inches='tight', transparent=True)
    plt.close('all')
    
def plot_directions(D, suffix):
    '''
    This function plots the distribution of initial replay directions.
    
    | **Args**
    | D:                            The distribution of initial replay directions.
    '''
    assert D.shape == (5, 5, 4)
    plt.figure(1, figsize=(10, 10))
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    for i in range(5):
        for j in range(5):
            plt.subplot(5, 5, i * 5 + j + 1)
            plt.bar(np.arange(4), D[i, j]/np.sum(D[i, j]), width=0.5)
            plt.ylim(0.0, 0.5)
            plt.xticks(np.arange(4), np.array([0, 90, 180, 270]))
            #plt.xticks(np.arange(4), np.array(['0°', '90°', '180°', '270°']))
            plt.yticks(np.array([0.0, 0.25, 0.5]), np.array([0.0, 0.25, 0.5]))
            ax = plt.gca()
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
    plt.savefig('plots/distribution_directions' + suffix + '.png', dpi=200, bbox_inches='tight', transparent=True)
    plt.savefig('plots/distribution_directions' + suffix + '.svg', dpi=200, bbox_inches='tight', transparent=True)
    plt.close('all')
    
def plot_reactivations(R, suffix=''):
    '''
    This function plots diffusion of replays.
    
    | **Args**
    | R:                            The distribution of starting positions.
    | bin_size:                     The bin size.
    '''
    plt.figure(1, figsize=(9, 2))
    plt.subplots_adjust(wspace=0.4)
    for i in range(4):
        plt.subplot(1, 4, i + 1)
        plt.title('$\Delta t=' + str(i + 1) + '$')
        if i == 0:
            plt.ylabel('y offset', fontsize=15)
        elif i == 1:
            plt.xlabel('x offset', fontsize=15, position=(1.2, 0))
        plt.pcolor(np.flip(R[:,:,i], axis=0), cmap='hot', vmin=0)
        #plt.pcolor(np.flip(np.sum(R[:,:,:(i+1)], axis=2), axis=0), cmap='hot')
        plt.xticks(np.array([0, int(R.shape[1]/2), R.shape[1]]), np.array([-int(R.shape[1]/2), 0, int(R.shape[1]/2)]))
        plt.yticks(np.array([0, int(R.shape[0]/2), R.shape[0]]), np.array([-int(R.shape[0]/2), 0, int(R.shape[0]/2)]))
    plt.savefig('plots/distribution_reactivation' + suffix + '.png', dpi=200, bbox_inches='tight', transparent=True)
    plt.savefig('plots/distribution_reactivation' + suffix + '.svg', dpi=200, bbox_inches='tight', transparent=True)
    plt.close('all')
    

if __name__ == "__main__":
    # params
    width, height = 100, 100
    occupancies = ['uniform', 'heterogeneous']
    modes = ['default', 'reverse']
    betas = [5, 10, 15]
    
    # analyze the distribution of replay starting positions and initial replay directions
    for occupancy in occupancies:
        for mode in modes:
            for beta in betas:
                replays = pickle.load(open('data/distribution_occupancy_' + occupancy + '_mode_' + mode + '_beta_' + str(beta) + '.pkl', 'rb'))
                starting_point_distribution = count_points(replays=replays, width=width, height=height, bin_size=1)
                replay_direction_distribution = compute_directions(replays=replays, width=width, height=height, bin_size=20, dir_bin_size=90)
                diffusion = compute_reactivations(replays=replays, width=width, height=height, view_width=25, view_height=25)
                H, H_max = compute_randomness(starting_point_distribution + 10**-60, width=width, height=height)
                plot_starting_distribution(starting_point_distribution, 1, '_' + occupancy + '_' + mode + '_beta_' + str(beta))
                plot_directions(replay_direction_distribution, '_' + occupancy + '_' + mode + '_beta_' + str(beta))
                plot_reactivations(diffusion, '_' + occupancy + '_' + mode + '_beta_' + str(beta))