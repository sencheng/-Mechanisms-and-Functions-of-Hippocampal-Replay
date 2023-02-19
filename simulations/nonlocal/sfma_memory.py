# basic imports
import numpy as np
# framework imports
from cobel.memory_modules.sfma_memory import SFMAMemory
  
    
class SFMAMemoryNonLocal(SFMAMemory):
    
    def __init__(self, interface_OAI, number_of_actions: int, gamma: float = 0.99, decay_inhibition: float = 0.9, decay_strength: float = 1., learning_rate: float = 0.9):
        '''
        Memory module to be used with the SFMA agent.
        Experiences are stored as a static table.
        
        Parameters
        ----------
        interface_OAI :                     The interface to the Open AI Gym environment.
        number_of_actions :                 The number of the agent's actions.
        gamma :                             The discount factor used to compute the successor representation or default representation.
        decay_inhibition :                  The factor by which inhibition is decayed.
        decay_strength :                    The factor by which the experience strengths are decayed.
        learning_rate :                     The learning rate with which experiences are updated.
        
        Returns
        ----------
        None
        '''
        super().__init__(interface_OAI, number_of_actions, gamma, decay_inhibition, decay_strength, learning_rate)
        # weighting of the current position during initialization
        self.weighting_local = 0.
    
    def replay(self, replay_length: int, current_state: int, current_action=None) -> list:
        '''
        This function replays experiences.
        
        Parameters
        ----------
        replay_length :                     The number of experiences that will be replayed.
        current_state :                     The state at which replay should start.
        current_action :                    The action with which replay should start.
        
        Returns
        ----------
        experiences :                       The replay batch.
        '''
        # initialize replay
        C = np.clip(self.C + self.weighting_local * np.tile(self.metric.D[current_state], self.number_of_actions), a_min=0, a_max=None)
        # we clip the strengths to catch negative values caused by rounding errors
        P = C/np.sum(C)
        exp = np.random.choice(np.arange(0, P.shape[0]), p=P)
        current_state = exp % self.number_of_states
        action = int(exp/self.number_of_states)
        next_state = self.states[current_state, action]
        # reset inhibition
        self.I *= 0
        # replay
        experiences = []
        for step in range(replay_length):
            # retrieve experience strengths
            C = np.copy(self.C)
            if self.C_normalize:
                C /= np.amax(C)
            # retrieve experience similarities
            D = np.tile(self.metric.D[current_state], self.number_of_actions)
            if self.D_normalize:
                D /= np.amax(D)
            if self.mode == 'forward':
                D = np.tile(self.metric.D[next_state], self.number_of_actions)
            elif self.mode == 'reverse':
                D = D[self.states.flatten(order='F')]
            elif self.mode == 'blend_forward':
                D += self.blend * np.tile(self.metric.D[next_state], self.number_of_actions)
            elif self.mode == 'blend_reverse':
                D += self.blend * D[self.states.flatten(order='F')]
            elif self.mode == 'interpolate':
                D = self.interpolation_fwd * np.tile(self.metric.D[next_state], self.number_of_actions) + self.interpolation_rev * D[self.states.flatten(order='F')]
            # retrieve inhibition
            I = np.tile(self.I, self.number_of_actions)
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
            action = int(exp/self.number_of_states)
            current_state = exp - (action * self.number_of_states)
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
