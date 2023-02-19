# basic imports
import numpy as np
import pickle


# params
number_of_states = 10000
number_of_actions = 4

# predefine occupancies
occupancies = {}
# uniform experience strength
occupancies['uniform'] = np.ones(number_of_states * number_of_actions)
# heterogeneous occupancies (here we let the strength fall of with distance to the environments center)
occupancies['heterogeneous'] = np.zeros(number_of_states * number_of_actions)
for state in range(number_of_states):
    x = int(state/100)
    y = state - 100 * x
    dist = np.sqrt(np.sum((np.array([50., 50.]) - np.array([x, y]))**2))
    occupancies['heterogeneous'][[state + action * number_of_states for action in range(number_of_actions)]] = -dist
occupancies['heterogeneous'] -= np.amin(occupancies['heterogeneous'])
occupancies['heterogeneous'] /= np.amax(occupancies['heterogeneous'])
occupancies['heterogeneous'] += 1.

# save
pickle.dump(occupancies, open('data/occupancies.pkl', 'wb'))
