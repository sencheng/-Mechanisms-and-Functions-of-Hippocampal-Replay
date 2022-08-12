# basic imports
import os
import numpy as np
import pickle


def recoverStates(replay):
    '''
    This function extracts the states of replayed experiences.
    
    | **Args**
    | replay:                       A sequence of replayed experiences.
    '''
    states = []
    for experience in replay:
        states += [experience['state']]
        
    return np.array(states)

def match_templates(replays):
    '''
    This function computes the fraction of replays that contained at least one state of the center.
    
    | **Args**
    | replays:                      Replays to be analyzed.
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
    fractions = {}
    fractions_seq = {}
    idx = {}
    for stepSize in replays:
        fraction, f_2, N, X = 0, 0, 0, 0
        idx[stepSize] = []
        for r, replay in enumerate(replays[stepSize]):
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
                    idx[stepSize] += [r]
                fraction += centerReplay
                f_2 += seq
        print(fraction, f_2, X)
        fractions[stepSize] = fraction# * 100 / f_2
        fractions_seq[stepSize] = f_2# * 100 / len(replays[stepSize])
        
    return fractions, fractions_seq, idx

def checkSequences(replays, sequences, idx, error_max=0.25):
    '''
    This function determines the parts of the environment which were replayed.
    
    | **Args**
    | replays:                      Replays to be analyzed.
    | sequences:                    The sequences which should be looked for.
    | error_max:                    Error threshold.
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

def checkShortcuts(seq, shortcuts, nonshortcuts):
    '''
    This function determines the fraction of shortcuts replayed.
    
    | **Args**
    | seq:                          Sequences found for each replay.
    | shortcuts:                    The shortcuts which should be looked for.
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
                    
    print(numberShortCuts, numberNonShortCuts, numberAll)
    
    
    return numberShortCuts, numberNonShortCuts, idx_shortcuts, numberAll