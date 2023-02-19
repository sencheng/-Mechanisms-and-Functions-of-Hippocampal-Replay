# basic imports
import numpy as np
import pickle


def recoverStates(replay: list) -> np.ndarray:
    '''
    This function extracts the states of replayed experiences.
    
    Parameters
    ----------
    replay :                            A sequence of replayed experiences.
    
    Returns
    ----------
    states :                            The recovered states.
    '''
    states = []
    for experience in replay:
        states += [experience['state']]
        
    return np.array(states)

def match_templates(replays) -> (int, int, list):
    '''
    This function computes the fraction of replays that contained at least one state of the center.
    
    Parameters
    ----------
    replays :                           Replays to be analyzed.
    
    Returns
    ----------
    fraction :                          The number of center replays.
    fraction_sequence :                 The number of detected sequences.
    idx :                               The indeces of replays with detected sequences.
    '''
    templates = {}
    templates['bwdCenter'] = np.array([16, 27, 38, 49, 60])
    templates['fwdCenter'] = np.flip(templates['bwdCenter'])
    templates['fwdTL'] = np.array([5, 4, 3, 2, 1, 0, 11, 22, 33])
    templates['bwdTL'] = np.flip(templates['fwdTL'])
    templates['fwdBL'] = np.array([33, 44, 55, 66, 67, 68, 69, 70, 71])
    templates['bwdBL'] = np.flip(templates['fwdBL'])
    templates['fwdTR'] = np.array([5, 6, 7, 8, 9, 10, 21, 32, 43])
    templates['bwdTR'] = np.flip(templates['fwdTR'])
    templates['fwdBR'] = np.array([43, 54, 65, 76, 75, 74, 73, 72, 71])
    templates['bwdBR'] = np.flip(templates['fwdBR'])
    fraction, fraction_seq, X = 0, 0, 0
    idx = []
    for r, replay in enumerate(replays):
        if len(replay) > 9:
            states = recoverStates(replay)
            centerReplay = False
            seq = False
            for template in templates:
                errors = []
                for i in range(states.shape[0] - templates[template].shape[0]):
                    error = np.sum(np.abs(states[i:i + templates[template].shape[0]] - templates[template]) > 0)
                    errors += [error]
                if np.amin(np.array(errors)) <= 2:
                    seq = True
                    X += 1
                    if template in ['bwdCenter', 'fwdCenter']:
                        centerReplay = True
            if seq:
                idx += [r]
            fraction += centerReplay
            fraction_seq += seq
        
    return fraction, fraction_seq, idx

def checkSequences(replays: list, sequences: dict, idx: list, error_max=0.25) -> list:
    '''
    This function determines the parts of the environment which were replayed.
    
    Parameters
    ----------
    replays :                           Replays to be analyzed.
    sequences :                         The sequences which should be looked for.
    idx :                               The indeces of the replays that will be analyzed.
    error_max :                         Match error threshold.
    
    Returns
    ----------
    results :                           The determined replay matches.
    '''
    results = []
    for r, replay in enumerate(replays):
        if r in idx:
            states = recoverStates(replay)
            errors = dict()
            seq = dict()
            for sequence in sequences:
                errors[sequence] = []
                for i in range(states.shape[0] - sequences[sequence].shape[0]):
                    error = np.sum(np.abs(sequences[sequence] - states[i:sequences[sequence].shape[0] + i]) > 0)
                    errors[sequence] += [error]
                errors[sequence] = np.array(errors[sequence])
                if np.amin(errors[sequence]) < error_max * sequences[sequence].shape[0]:
                    seq[np.argmin(errors[sequence])] = sequence
            results += [seq]
        
    return results

def checkShortcuts(seq: list, shortcuts: dict, nonshortcuts: dict) -> (int, int, list, int):
    '''
    This function determines the fraction of shortcuts replayed.
    
    Parameters
    ----------
    seq :                               Sequences found for each replay.
    shortcuts :                         The shortcuts which should be looked for.
    nonshortcuts :                      The non-shortcuts which should be looked for.
    
    Returns
    ----------
    number_shortcuts :                  The number of shortcuts.
    number_non_shortcuts :              The number of non-shortcuts.
    idx_shortcuts :                     The indeces of replays containing shortcut replays.
    number_all :                        The number of all sequences.
    '''
    numberShortCuts, numberNonShortCuts, numberAll = 0, 0, 0
    idx_shortcuts = []
    
    for r, replay in enumerate(seq):
        keys = sorted(list(replay.keys()))
        if len(keys) > 1:
            sc = []
            idx =  []
            for key in keys:
                sc += [replay[key]]
                idx += [key]
            for i in range(len(sc) - 1):
                dist = np.abs(np.abs(idx[i + 1] - idx[i]) - 8)
                if dist < 3:
                    numberAll += 1
                    if (sc[i], sc[i + 1]) in shortcuts:
                        numberShortCuts += 1
                        idx_shortcuts += [r]
                    if (sc[i], sc[i + 1]) in nonshortcuts:
                        numberNonShortCuts += 1    
    
    return numberShortCuts, numberNonShortCuts, idx_shortcuts, numberAll


def analyze(sequences, shortcuts, nonshortcuts, prefix='') -> dict:
    '''
    This function analyses the occurence of shortcut replays.
    
    Parameters
    ----------
    sequences :                         Sequences templates used for matching.
    shortcuts :                         The shortcuts which should be looked for.
    nonshortcuts :                      The non-shortcuts which should be looked for.
    prefix :                            The prefix of the files that will be analyzed.
    
    Returns
    ----------
    short_cuts :                        The results of the analysis.
    '''
    betas = np.linspace(5, 15, 11)
    modes = ['default', 'reverse']
    use_recency = [True, False]
    
    short_cuts = {}
    
    for mode in modes:
        short_cuts[mode] = {}
        for recency in use_recency:
            short_cuts[mode][recency] = {}
            for beta in betas:
                short_cuts[mode][recency][beta] = {}
                fileName = 'data/static/' + prefix + 'mode_' + mode + '_recency_' + ('Y' if recency else 'N') + '_beta_' + str(beta) + '.pkl'
                simulation_data = pickle.load(open(fileName, 'rb'))
                for condition in simulation_data:
                    results = {'center': [], 'shortcuts': [], 'non_shortcuts': [], 'idx_shortcuts': []}
                    for trial in simulation_data[condition]:
                        fraction, fraction_seq, idx = match_templates(trial)
                        sequs = checkSequences(trial, sequences, idx)
                        sc, numberNonShortCuts, idx_shortcuts, numberAll = checkShortcuts(sequs, shortcuts, nonshortcuts)
                        results['center'] +=  [fraction]
                        results['shortcuts'] += [sc]
                        results['non_shortcuts'] += [numberNonShortCuts]
                        results['idx_shortcuts'] += [idx_shortcuts]
                    short_cuts[mode][recency][beta][condition] = results
                    
    return short_cuts


if __name__ == '__main__':
    # define possible replay sequences
    sequences = dict()
    # define corner replays
    sequences['fwdTL'] = np.array([5, 4, 3, 2, 1, 0, 11, 22, 33])
    sequences['bwdTL'] = np.flip(sequences['fwdTL'])
    sequences['fwdBL'] = np.array([33, 44, 55, 66, 67, 68, 69, 70, 71])
    sequences['bwdBL'] = np.flip(sequences['fwdBL'])
    sequences['fwdTR'] = np.array([5, 6, 7, 8, 9, 10, 21, 32, 43])
    sequences['bwdTR'] = np.flip(sequences['fwdTR'])
    sequences['fwdBR'] = np.array([43, 54, 65, 76, 75, 74, 73, 72, 71])
    sequences['bwdBR'] = np.flip(sequences['fwdBR'])
    
    # define shortcuts
    shortcuts = [('bwdTR', 'fwdTL'), ('bwdTL', 'fwdTR'), ('fwdBR', 'bwdBL'), ('fwdBL', 'bwdBR')]
    # define nonshortcuts
    nonshortcuts = [('bwdTR', 'fwdTR'), ('bwdTL', 'fwdTL'), ('fwdTR', 'bwdTR'), ('fwdTL', 'bwdTL'),
                    ('bwdBR', 'fwdBR'), ('bwdBL', 'fwdBL'), ('fwdBR', 'bwdBR'), ('fwdBL', 'bwdBL')]
    
    short_cuts = analyze(sequences, shortcuts, nonshortcuts)
    pickle.dump(short_cuts, open('analysis/shortCuts_static.pkl', 'wb'))
