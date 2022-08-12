# basic imports
import numpy as np
# framework imports
from cobel.memory_modules.memory_utils.metrics import DR
  
    
class SFMAMemory():
    '''
    Memory module to be used with the SFMA agent.
    Experiences are stored as a static table.
    
    | **Args**
    | interfaceOAI:                 The Open AI interface.
    | numberOfActions:              The number of the agent's actions.
    | gamma:                        The discount factor used to compute the successor representation or default representation.
    | decay_inhibition:             The factor by which inhibition is decayed.
    | decay_strength:               The factor by which the experience strengths are decayed.
    | learningRate:                 The learning rate with which experiences are updated.
    '''
    
    def __init__(self, interfaceOAI, numberOfActions, gamma=0.99, decay_inhibition=0.9, decay_strength=1., learningRate=0.9):
        # initialize variables
        self.numberOfStates = interfaceOAI.world['states']
        self.numberOfActions = numberOfActions
        self.decay_inhibition = decay_inhibition
        self.decay_strength = decay_strength
        self.decay_recency = 0.9
        self.learningRate = learningRate
        self.beta = 20
        self.rlAgent = None
        # experience strength modulation parameters
        self.reward_mod_local = False # increase during experience
        self.error_mod_local = False # increase during experience
        self.reward_mod = False # increase during experience
        self.error_mod = False # increase during experience
        self.policy_mod = False # added before replay
        self.state_mod = False # 
        # similarity metric
        self.metric = DR(interfaceOAI, gamma)
        # prepare memory structures
        self.rewards = np.zeros((self.numberOfStates, self.numberOfActions))
        self.states = np.tile(np.arange(self.numberOfStates).reshape(self.numberOfStates, 1), self.numberOfActions).astype(int)
        self.terminals = np.zeros((self.numberOfStates, self.numberOfActions)).astype(int)
        # prepare replay-relevant structures
        self.C = np.zeros(self.numberOfStates * self.numberOfActions) # strength
        self.T = np.zeros(self.numberOfStates * self.numberOfActions) # recency
        self.I = np.zeros(self.numberOfStates) # inhibition
        # increase step size
        self.C_step = 1.
        self.I_step = 1.
        # priority rating threshold
        self.R_threshold = 10**-6
        # always reactive experience with highest priority rating
        self.deterministic = False
        # consider recency of experience
        self.recency = False
        # normalize variables
        self.C_normalize = False
        self.D_normalize = False
        self.R_normalize = True
        # replay mode
        self.mode = 'default'
        # modulates reward
        self.reward_modulation = 1.
        # weighting of forward/reverse mode when using blending modes
        self.blend = 0.1
        # weightings of forward abdreverse modes when using interpolation mode
        self.interpolation_fwd, self.interpolation_rev = 0.5, 0.5
        
    def store(self, experience):
        '''
        This function stores a given experience.
        
        | **Args**
        | experience:                   The experience to be stored.
        '''
        state, action = experience['state'], experience['action']
        # update experience
        self.rewards[state][action] += self.learningRate * (experience['reward'] - self.rewards[state][action])
        self.states[state][action] = experience['next_state']
        self.terminals[state][action] = experience['terminal']
        # update replay-relevent structures
        self.C *= self.decay_strength
        self.C[self.numberOfStates * action + state] += self.C_step
        self.T *= self.decay_recency
        self.T[self.numberOfStates * action + state] = 1.
        # local reward modulation (affects this experience only)
        if self.reward_mod_local:
            self.C[self.numberOfStates * action + state] += experience['reward'] * self.reward_modulation
        # reward modulation (affects close experiences)
        if self.reward_mod:
            modulation = np.tile(self.metric.D[experience['state']], self.numberOfActions)
            self.C += experience['reward'] * modulation * self.reward_modulation
        # local RPE modulation (affects this experience only)
        if self.error_mod_local:
            self.C[self.numberOfStates * action + state] += np.abs(experience['error'])
        # RPE modulation (affects close experiences)
        if self.error_mod:
            modulation = np.tile(self.metric.D[experience['next_state']], self.numberOfActions)
            self.C += np.abs(experience['error']) * modulation
        # additional strength increase of all experiences at current state
        if self.state_mod:
            self.C[[state + self.numberOfStates * a for a in range(self.numberOfActions)]] += 1.
    
    def replay(self, replayLength, current_state=None, current_action=None):
        '''
        This function replays experiences.
        
        | **Args**
        | replayLength:                 The number of experiences that will be replayed.
        | current_state:                State at which replay should start.
        | current_action:               Action with which replay should start.
        '''
        action = current_action
        # if no action is specified pick one at random
        if current_action is None:
            action = np.random.randint(self.numberOfActions)
        # if a state is not defined, then choose an experience according to relative experience strengths
        if current_state is None:
            # we clip the strengths to catch negative values caused by rounding errors
            P = np.clip(self.C, a_min=0, a_max=None)/np.sum(np.clip(self.C, a_min=0, a_max=None))
            exp = np.random.choice(np.arange(0, P.shape[0]), p=P)
            current_state = exp % self.numberOfStates
            action = int(exp/self.numberOfStates)
        next_state = self.states[current_state, action]
        # reset inhibition
        self.I *= 0
        # replay
        experiences = []
        for step in range(replayLength):
            # retrieve experience strengths
            C = np.copy(self.C)
            if self.C_normalize:
                C /= np.amax(C)
            # retrieve experience similarities
            D = np.tile(self.metric.D[current_state], self.numberOfActions)
            if self.D_normalize:
                D /= np.amax(D)
            if self.mode == 'forward':
                D = np.tile(self.metric.D[next_state], self.numberOfActions)
            elif self.mode == 'reverse':
                D = D[self.states.flatten(order='F')]
            elif self.mode == 'blend_forward':
                D += self.blend * np.tile(self.metric.D[next_state], self.numberOfActions)
            elif self.mode == 'blend_reverse':
                D += self.blend * D[self.states.flatten(order='F')]
            elif self.mode == 'interpolate':
                D = self.interpolation_fwd * np.tile(self.metric.D[next_state], self.numberOfActions) + self.interpolation_rev * D[self.states.flatten(order='F')]
            # retrieve inhibition
            I = np.tile(self.I, self.numberOfActions)
            # compute priority ratings
            R = C * D * (1 - I)
            if self.recency:
                R *= self.T
            # apply threshold to priority ratings
            R[R < self.R_threshold] = 0.
            # stop replay sequence if all priority ratings are all zero
            if np.sum(R) == 0.:
                break
            # determine state and action
            if self.R_normalize:
                R /= np.amax(R)
            exp = np.argmax(R)
            if not self.deterministic:
                # compute activation probabilities
                probs = self.softmax(R, -1, self.beta)
                probs = probs/np.sum(probs)
                exp = np.random.choice(np.arange(0,probs.shape[0]), p=probs)
            # determine experience tuple
            action = int(exp/self.numberOfStates)
            current_state = exp - (action * self.numberOfStates)
            next_state = self.states[current_state][action]
            # apply inhibition
            self.I *= self.decay_inhibition
            self.I[current_state] = min(self.I[current_state] + self.I_step, 1.)
            # "reactivate" experience
            experience = {'state': current_state, 'action': action, 'reward': self.rewards[current_state][action],
                          'next_state': next_state, 'terminal': self.terminals[current_state][action]}
            experiences += [experience]
            # stop replay at terminal states
            #if experience['terminal']:
            #    break
            
        return experiences
    

    def softmax(self, data, offset=0, beta=5):
        '''
        This function computes the softmax over the input.
        
        | **Args**
        | data:                         Input of the softmax function.
        | offset:                       Offset added after applying the softmax function.
        | beta:                         Beta value.
        '''
        exp = np.exp(data * beta) + offset
        if np.sum(exp) == 0:
            exp.fill(1)
        else:
            exp /= np.sum(exp)
        
        return exp
    
    def retrieve_random_batch(self, numberOfExperiences, mask):
        '''
        This function retrieves a number of random experiences.
        
        | **Args**
        | numberOfExperiences:          The number of random experiences to be drawn.
        '''
        # draw random experiences
        probs = np.ones(self.numberOfStates * self.numberOfActions) * mask.astype(int)
        probs /= np.sum(probs)
        idx = np.random.choice(np.arange(self.numberOfStates * self.numberOfActions), numberOfExperiences, p=probs)
        # determine indeces
        idx = np.unravel_index(idx, (self.numberOfStates, self.numberOfActions), order='F')
        # build experience batch
        experiences = []
        for exp in range(numberOfExperiences):
            state, action = idx[0][exp], idx[1][exp]
            experiences += [{'state': state, 'action': action, 'reward': self.rewards[state][action],
                             'next_state': self.states[state][action], 'terminal': self.terminals[state][action]}]
            
        return experiences