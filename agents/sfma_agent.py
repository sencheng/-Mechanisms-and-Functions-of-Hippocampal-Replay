# basic imports
import numpy as np
# framework imports
from cobel.agents.rl_agent import callbacks
from cobel.agents.dyna_q_agent import AbstractDynaQAgent
# custom imports
from memory_modules.sfma_memory import SFMAMemory

    
class SFMAAgent(AbstractDynaQAgent):
    '''
    Implementation of a Dyna-Q agent using the Spatial Structure and Frequency-weighted Memory Access (SMA) memory module.
    Q-function is represented as a static table.
    
    | **Args**
    | interface_OAI:                The interface to the Open AI Gym environment.
    | epsilon:                      The epsilon value for the epsilon greedy policy.
    | learning_rate:                The learning rate with which the Q-function is updated.
    | gamma:                        The discount factor used to compute the TD-error.
    | gamma_SR:                     The discount factor used by the SMA memory module.
    | custom_callbacks:             The custom callbacks defined by the user.
    '''
    
    class callbacksSFMA(callbacks):
        '''
        Callback class. Used for visualization and scenario control.
        
        | **Args**
        | rl_parent:                    Reference to the RL agent.
        | custom_callbacks:             The custom callbacks defined by the user.
        '''
        
        def __init__(self, rl_parent, custom_callbacks={}):
            super().__init__(rl_parent, custom_callbacks)
                    
        def on_replay_begin(self, logs):
            '''
            The following function is called whenever experiences are replayed.
            
            | **Args**
            | logs:                         The trial log.
            '''
            logs['rl_parent'] = self.rl_parent
            if 'on_replay_begin' in self.custom_callbacks:
                for custom_callback in self.custom_callbacks['on_replay_begin']:
                    custom_callback(logs)
        
        def on_replay_end(self, logs):
            '''
            The following function is called after experiences were replayed.
            
            | **Args**
            | logs:                         The trial log.
            '''
            logs['rl_parent'] = self.rl_parent
            if 'on_replay_end' in self.custom_callbacks:
                for custom_callback in self.custom_callbacks['on_replay_end']:
                    custom_callback(logs)
                
            
    def __init__(self, interface_OAI, epsilon=0.3, beta=5, learning_rate=0.9, gamma=0.99, gamma_SR=0.99, custom_callbacks={}):
        super().__init__(interface_OAI, epsilon=epsilon, beta=beta, learning_rate=learning_rate, gamma=gamma)
        # Q-table
        self.Q = np.zeros((self.number_of_states, self.number_of_actions))
        # memory module
        self.M = SFMAMemory(self.interface_OAI, self.number_of_actions, gamma_SR)
        # set up the visualizer for the RL agent behavior/reward outcome
        self.engaged_callbacks = self.callbacksSFMA(self, custom_callbacks)
        # training
        self.replays_per_trial = 1 # number of replay batches
        self.random_replay = False # if true, random replay batches are sampled
        self.dynamic_mode = False # if true, the replay mode is determined by the cumulative td-error
        self.offline = False # if true, the agent learns only with experience replay
        self.start_replay = False # if true, a replay trace is generated at the start of each trial
        self.td = 0. # stores the temporal difference errors accounted during each trial
        
    def train(self, number_of_trials=100, max_number_of_steps=50, replay_batch_size=100, no_replay=False):
        '''
        This function is called to train the agent.
        
        | **Args**
        | number_of_trials:             The number of trials the Dyna-Q agent is trained.
        | max_number_of_steps:          The maximum number of steps per trial.
        | replay_batch_size:            The number of random that will be replayed.
        | no_replay:                    If true, experiences are not replayed.
        '''
        for trial in range(number_of_trials):
            # prepare trial log
            trial_log = {'trial_reward': 0, 'trial': self.current_trial, 'trial_session': trial, 'steps': 0, 'replay_mode': self.M.mode}
            # callback
            self.engaged_callbacks.on_trial_begin(trial_log)
            # reset environment
            state = self.interface_OAI.reset()
            if self.start_replay:
                self.engaged_callbacks.on_replay_begin(trial_log)
                trial_log['replay'] = self.M.replay(replay_batch_size, state)
                self.engaged_callbacks.on_replay_end(trial_log)
            for step in range(max_number_of_steps):
                # determine next action
                action = self.select_action(state, self.epsilon, self.beta)
                # perform action
                next_state, reward, end_trial, callback_value = self.interface_OAI.step(action)
                # make experience
                experience = {'state': state, 'action': action, 'reward': reward, 'next_state': next_state, 'terminal': (1 - end_trial)}
                # update Q-function with experience
                self.update_Q(experience, self.offline)
                # store experience
                self.M.store(experience)
                # log reward
                trial_log['trial_reward'] += reward
                # update current state
                state = next_state
                # stop trial when the terminal state is reached
                if end_trial:
                    break
            self.current_trial += 1
            # log steps
            trial_log['steps'] = step
            # perform experience replay
            if not no_replay:
                # determine replay mode if modes are chosen dynamically
                if self.dynamic_mode:
                    p_mode = 1 / (1 + np.exp(-(self.td * 5 - 2)))
                    trial_log['replay_mode'] = ['reverse', 'default'][np.random.choice(np.arange(2), p=np.array([p_mode, 1 - p_mode]))]
                    self.M.mode = trial_log['replay_mode']
                    self.td = 0.
                # replay
                for i in range(self.replays_per_trial):
                    self.engaged_callbacks.on_replay_begin(trial_log)
                    trial_log['replay'] = self.replay(replay_batch_size, next_state)
                    self.engaged_callbacks.on_replay_end(trial_log)
                self.M.T.fill(0)
            # callback
            self.engaged_callbacks.on_trial_end(trial_log)
            
    def test(self, number_of_trials=100, max_number_of_steps=50):
        '''
        This function is called to test the agent.
        
        | **Args**
        | number_of_trials:             The number of trials the Dyna-Q agent is trained.
        | max_number_of_steps:          The maximum number of steps per trial.
        '''
        for trial in range(number_of_trials):
           # prepare trial log
            trial_log = {'trial_reward': 0, 'trial': self.current_trial, 'trial_session': trial, 'steps': 0, 'replay_mode': self.M.mode}
            # callback
            self.engaged_callbacks.on_trial_begin(trial_log)
            # reset environment
            state = self.interface_OAI.reset()
            for step in range(max_number_of_steps):
                # determine next action
                action = self.select_action(state, self.epsilon, self.beta)
                # perform action
                next_state, reward, end_trial, callback_value = self.interface_OAI.step(action)
                # log reward
                trial_log['trial_reward'] += reward
                # update current state
                state = next_state
                # stop trial when the terminal state is reached
                if end_trial:
                    break
            self.current_trial += 1
            # log steps
            trial_log['steps'] = step
            # callback
            self.engaged_callbacks.on_trial_end(trial_log)
            
    def replay(self, replay_batch_size=200, state=None):
        '''
        This function replays experiences to update the Q-function.
        
        | **Args**
        | replay_batch_size:            The number of random that will be replayed.
        | state:                        The state at which replay should be initiated.
        '''
        # sample batch of experiences
        replay_batch = []
        if self.random_replay:
            mask = np.ones((self.number_of_states * self.number_of_actions))
            if self.mask_actions:
                mask = np.copy(self.action_mask).flatten(order='F')
            replay_batch = self.M.retrieve_random_batch(replay_batch_size, mask)
        else:
            replay_batch = self.M.replay(replay_batch_size, state)
        # update the Q-function with each experience
        for experience in replay_batch:
            self.update_Q(experience)
            
        return replay_batch
    
    def update_Q(self, experience, no_update=False):
        '''
        This function updates the Q-function with a given experience.
        
        | **Args**
        | experience:                   The experience with which the Q-function will be updated.
        | no_update:                    If true, the Q-function is not updated.
        '''
        # make mask
        mask = np.arange(self.number_of_actions)
        if self.mask_actions:
            mask = self.action_mask[experience['next_state']]
        # compute TD-error
        td = experience['reward']
        td += self.gamma * experience['terminal'] * np.amax(self.retrieve_Q(experience['next_state'])[mask])
        td -= self.retrieve_Q(experience['state'])[experience['action']]
        # update Q-function with TD-error
        if not no_update:
            self.Q[experience['state']][experience['action']] += self.learning_rate * td
        # store temporal difference error
        experience['error'] = td
        #if self.dynamic_mode:
        self.td += np.abs(td)
        
        return td
    
    def retrieve_Q(self, state):
        '''
        This function retrieves Q-values for a given state.
        
        | **Args**
        | state:                        The state for which Q-values should be retrieved.
        '''
        # retrieve Q-values, if entry exists
        return self.Q[state]
    
    def predict_on_batch(self, batch):
        '''
        This function retrieves Q-values for a batch of states.
        
        | **Args**
        | batch:                        The batch of states.
        '''
        predictions = []
        for state in batch:
            predictions += [self.retrieve_Q(state)]
            
        return np.array(predictions)
