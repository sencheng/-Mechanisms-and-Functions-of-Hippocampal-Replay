# basic imports
import numpy as np
# tensorflow imports
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import Dense
# framework imports
from cobel.agents.dyna_q_agent import AbstractDynaQAgent
# custom imports
from memory_modules.sfma_memory import SFMAMemory


class DeepSFMAAgent(AbstractDynaQAgent):
    '''
    Implementation of a Dyna-Q agent using the Spatial Structure and Frequency-weighted Memory Access (SMA) memory module.
    Q-function is represented as a deep Q-network (DQN).
    
    | **Args**
    | interface_OAI:                The interface to the Open AI Gym environment.
    | epsilon:                      The epsilon value for the epsilon greedy policy.
    | learning_rate:                The learning rate with which the Q-function is updated.
    | gamma:                        The discount factor used to compute the TD-error.
    | gamma_SR:                     The discount factor used by the SMA memory module.
    | observations:                 The set of observations that will be mapped to environmental states. If undefined, a one-hot encoding for will be generated.
    | model:                        The model that will be used to represent the Q-function. If undefined, a small fully-connected network will be build.
    | custom_callbacks:             The custom callbacks defined by the user.
    '''                
            
    def __init__(self, interface_OAI, epsilon=0.3, beta=5, learning_rate=0.9, gamma=0.99, gamma_SR=0.99, observations=None, model=None, custom_callbacks={}):
        super().__init__(interface_OAI, epsilon=epsilon, beta=beta, learning_rate=learning_rate, gamma=gamma, custom_callbacks=custom_callbacks)
        # observations
        self.observations = observations
        if self.observations is None:
            self.observations = np.eye(self.number_of_states)
        # model
        self.model = self.build_model(model)
        # memory module
        self.M = SFMAMemory(self.interface_OAI, self.number_of_actions, gamma_SR)
        # training
        self.replays_per_trial = 1 # number of replay batches
        self.random_replay = False # if true, random replay batches are sampled
        self.dynamic_mode = False # if true, the replay mode is determined by the cumulative td-error
        self.td = 0. # stores the temporal difference errors accounted during each trial
        self.target_model_update = 10**-2 # target model blending factor
        self.updates_per_replay = 1 # number of BP updates per replay batch
        self.local_targets = True # if true, the model will be updated with locally computed target values
        self.randomize_subsequent_replays = False # if true, only the first replay after each trial uses SFMA
        
    def build_model(self, model=None):
        '''
        This function builds the DQN's target and online models.
        
        | **Args**
        | model:                        The DNN model to be used by the agent. If None, a small dense DNN is created by default.
        '''
        # build target model
        if model is None:
            self.model_target = Sequential()
            self.model_target.add(Dense(units=64, input_shape=(self.observations.shape[1],), activation='tanh'))
            self.model_target.add(Dense(units=64, activation='tanh'))
            self.model_target.add(Dense(units=self.number_of_actions, activation='linear'))
        else:
            self.model_target = model
        self.model_target.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
        # build online model by cloning the target model
        self.model_online = clone_model(self.model_target)
        self.model_online.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

        
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
            for step in range(max_number_of_steps):
                # determine next action
                action = self.select_action(state, self.epsilon, self.beta)
                # perform action
                next_state, reward, stop_episode, callback_value = self.interface_OAI.step(action)
                # make experience
                experience = {'state': state, 'action': action, 'reward': reward, 'next_state': next_state, 'terminal': (1 - stop_episode)}
                # store experience
                self.M.store(experience)
                # log behavior and reward
                trial_log['trial_reward'] += reward
                # update current state
                state = next_state
                # stop trial when the terminal state is reached
                if stop_episode:
                    break
            self.current_trial += 1
            # log steps
            trial_log['steps'] = step
            # perform experience replay
            if not no_replay:
                # determine replay mode if modes are chosen dynamically
                if self.dynamic_mode:
                    p_mode = 1 / (1 + np.exp(-(self.td * 5 - 2)))
                    self.M.mode = ['reverse', 'default'][np.random.choice(np.arange(2), p=np.array([p_mode, 1 - p_mode]))]
                    self.td = 0.
                # replay
                replays = []
                for i in range(self.replays_per_trial):
                    if i > 0 and self.randomize_subsequent_replays:
                        replays.append(self.replay(replay_batch_size, next_state, True))
                    else:
                        replays.append(self.replay(replay_batch_size, next_state, self.random_replay))
                # update model with generated replay
                if self.local_targets:
                    self.update_local(replays)
                else:
                    self.update_step_wise(replays)
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
                next_state, reward, stop_episode, callback_value = self.interface_OAI.step(action)
                # log behavior and reward
                trial_log['trial_reward'] += reward
                # update current state
                state = next_state
                # stop trial when the terminal state is reached
                if stop_episode:
                    break
            self.current_trial += 1
            # log steps
            trial_log['steps'] = step
            # callback
            self.engaged_callbacks.on_trial_end(trial_log)
            
    def replay(self, replay_batch_size=200, state=None, random_replay=False):
        '''
        This function replays experiences to update the Q-function.
        
        | **Args**
        | replay_batch_size:            The number of random that will be replayed.
        | state:                        The state at which replay should be initiated.
        '''
        # sample batch of experiences
        replay_batch = []
        if random_replay:
            mask = np.ones((self.number_of_states * self.number_of_actions))
            if self.mask_actions:
                mask = np.copy(self.action_mask).flatten(order='F')
            replay_batch = self.M.retrieve_random_batch(replay_batch_size, mask)
        else:
            replay_batch = self.M.replay(replay_batch_size, state)
            
        return replay_batch
    
    def update_local(self, replays):
        '''
        This function locally computes new target values to update the model.
        
        | **Args**
        | replays:                      The replays with which the model will be updated.
        '''
        for replay in replays:
            # compute local Q-function
            Q_local = self.model_target.predict_on_batch(self.observations)
            # update local Q-function
            batch_states = []
            for experience in replay:
                batch_states.append(experience['state'])
                # prepare action mask
                mask = np.ones(self.number_of_actions).astype(bool)
                if self.mask_actions:
                    mask = self.action_mask[experience['next_state']]
                # compute target
                target = experience['reward']
                target += self.gamma * experience['terminal'] * np.amax(Q_local[experience['next_state']][mask])
                # update local Q-function
                Q_local[experience['state']][experience['action']] = target
            # update model
            self.update_model(self.observations[batch_states], Q_local[batch_states], self.updates_per_replay)
            
    def update_step_wise(self, replays):
        '''
        This function updates the model step-wise on all sequences.
        
        | **Args**
        | replays:                      The replays with which the model will be updated.
        '''
        # update step-wise
        for step in range(len(replays[0])):
            states, targets = [], []
            Q_local = self.model_target.predict_on_batch(self.observations)
            for replay in replays:
                # recover variables from experience
                states.append(replay[step]['state'])
                reward, action = replay[step]['reward'], replay[step]['action']
                terminal, next_state = replay[step]['terminal'], replay[step]['next_state']
                # prepare action mask
                mask = np.ones(self.number_of_actions).astype(bool)
                if self.mask_actions:
                    mask = self.action_mask[next_state]
                # compute target
                target = Q_local[replay[step]['state']]
                target[action] = reward + self.gamma * terminal * np.amax(Q_local[next_state][mask])
                targets.append(target)
            # update model
            self.update_model(self.observations[states], np.array(targets), self.updates_per_replay)
            
    def update_model(self, observations, targets, number_of_updates=1):
        '''
        This function updates the model on a batch of experiences.
        
        | **Args**
        | observations:                 The observations.
        | targets:                      The targets.
        | number_of_updates:            The number of backpropagation updates that should be performed on this batch.
        '''
        # update online model
        for update in range(number_of_updates):
            self.model_online.train_on_batch(observations, targets)
        # update target model by blending it with the online model
        weights_target = np.array(self.model_target.get_weights(), dtype=object)
        weights_online = np.array(self.model_online.get_weights(), dtype=object)
        weights_target += self.target_model_update * (weights_online - weights_target)
        self.model_target.set_weights(weights_target)
    
    def update_Q(self, experience):
        '''
        This function is a dummy function and does nothing (implementation required by parent class).
        
        | **Args**
        | experience:                   The experience with which the Q-function will be updated.
        '''
        pass
    
    def retrieve_Q(self, state):
        '''
        This function retrieves Q-values for a given state.
        
        | **Args**
        | state:                        The state for which Q-values should be retrieved.
        '''
        # retrieve Q-values, if entry exists
        return self.model_online.predict_on_batch(np.array([self.observations[state]]))[0]
    
    def predict_on_batch(self, batch):
        '''
        This function retrieves Q-values for a batch of states.
        
        | **Args**
        | batch:                        The batch of states.
        '''  
        return self.model_online.predict_on_batch(self.observations[batch])
