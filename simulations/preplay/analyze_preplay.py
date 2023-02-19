# basic imports
import numpy as np
import pickle


def recover_states(replay: list) -> np.ndarray:
    '''
    This function extracts the states of replayed experiences.
    
    Parameters
    ----------
    replay :                            A sequence of replayed experiences.
    
    Returns
    ----------
    states :                            The recovered sequence of replayed states.
    '''
    states = []
    for experience in replay:
        states += [experience['state']]
        
    return np.array(states)

def analyze_replays(replays: list, templates: dict, error_max=0.4) -> dict:
    '''
    This function determines the parts of the environment which were replayed.
    
    Parameters
    ----------
    replays :                           A list of replays.
    templates :                         The replay sequences which should be looked for.
    error_max :                         The maximum fraction of template mismatches.
    
    Returns
    ----------
    matches :                           Dictionary containing the number of matches for each template.
    '''
    matches = {'cued': 0, 'uncued': 0, 'stem': 0, 'fwdCued': 0, 'bwdCued': 0}
    for replay in replays:
        if len(replay) > 0:
            states = recover_states(replay)
            for template in templates:
                kernel = np.copy(templates[template])
                signal = np.copy(states)
                if kernel.shape[0] > signal.shape[0]:
                    kernel = np.copy(states)
                    signal = np.copy(templates[template])
                errors = []
                for i in range(signal.shape[0] - kernel.shape[0] + 1):
                    error = np.sum(np.abs(signal[i:kernel.shape[0] + i] - kernel) > 0)
                    errors += [error]
                errors = np.array(errors)
                if errors[np.argmin(errors)] <= error_max * kernel.shape[0]:
                    if template in ['fwdCued', 'bwdCued']:
                        matches['cued'] += 1
                        matches[template] += 1
                    elif template in ['fwdUncued', 'bwdUncued']:
                        matches['uncued'] += 1
                    elif template in ['fwdStem', 'bwdStem']:
                        matches['stem'] += 1
                        
    return matches

def compute_reactivation_map(replays: list) -> np.ndarray:
    '''
    This function computes the reactivation probabilities for all states.
    
    Parameters
    ----------
    replays :                           A list of replays.
    
    Returns
    ----------
    R :                                 The computed reactivation map.
    '''
    R = np.zeros((10, 7))
    for replay in replays:
        states = recover_states(replay)
        for state in states:
            height = int(state/7)
            width = state - height * 7
            R[height, width] += 1
            
    return R/np.amax(R)
    

if __name__ == "__main__":
    # define templates
    templates = {}
    templates['fwdCued'] = np.array([17, 10, 3, 4, 5, 6])
    templates['fwdUncued'] = np.array([17, 10, 3, 2, 1, 0])
    templates['fwdStem'] = np.array([66, 59, 52, 45, 38, 31, 24])
    templates['bwdCued'] = np.flip(templates['fwdCued'])
    templates['bwdUncued'] = np.flip(templates['fwdUncued'])
    templates['bwdStem'] = np.flip(templates['fwdStem'])
    
    # params    
    attention_cued = np.array([0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0])
    attention_arms = np.array([0.0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2])
    modes = ['default', 'reverse']
    gammas = [0.1, 0.8]
    
    #
    for gamma in gammas:
        for mode in modes:
            print('Discount factor: ' + str(gamma) + ', Mode: ' + mode)
            R_cued, R_dir = np.zeros((11, 11)), np.zeros((11, 11))
            for i, cued in enumerate(attention_cued):
                for j, arms in enumerate(attention_arms):
                    file_name = 'data/sweep/gamma_' + str(gamma) + '_mode_' + mode + '_arms_' + str(arms) + '_cued_' + str(cued) + '.pkl'
                    replays = pickle.load(open(file_name, 'rb'))
                    matches = analyze_replays(replays, templates)
                    R_cued[i, j] = 100 * matches['cued']/(matches['cued'] + matches['stem'] + matches['uncued'])
                    if (matches['fwdCued'] + matches['bwdCued']) > 0:
                        R_dir[i, j] = 100 * matches['fwdCued']/(matches['fwdCued'] + matches['bwdCued'])
            # store fractions
            pickle.dump({'preplay': R_cued, 'fwd': R_dir, 'fractions': attention_cued, 'R': attention_arms},
                        open('data/analyzed_preplay_gamma_' + str(gamma) + '_mode_' + mode + '.pkl', 'wb'))
