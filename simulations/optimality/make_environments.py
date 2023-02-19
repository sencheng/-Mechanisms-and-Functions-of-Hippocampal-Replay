# basic imports
import pickle
import numpy as np
# framework imports
from cobel.misc.gridworld_tools import make_open_field, make_gridworld


# make linear track environment
linear_track = make_open_field(1, 10, goal_state=0, reward=1)
linear_track['starting_states'] = np.array([9])
pickle.dump(linear_track, open('environments/linear_track.pkl', 'wb'))

# make open field environment
open_field = make_open_field(10, 10, goal_state=0, reward=1)
open_field['starting_states'] = np.array([55])
pickle.dump(open_field, open('environments/open_field.pkl', 'wb'))

# make labyrinth environment
invalidTransitions = [(1, 2), (11, 12), (21, 22), (31, 32), (41, 42), (51, 52), (61, 62), (71, 72)]
invalidTransitions += [(2, 1), (12, 11), (22, 21), (32, 31), (42, 41), (52, 51), (62, 61), (72, 71)]
invalidTransitions += [(14, 24), (15, 25), (16, 26), (17, 27)]
invalidTransitions += [(24, 14), (25, 15), (26, 16), (27, 17)]
invalidTransitions += [(23, 24), (33, 34), (43, 44), (53, 54), (63, 64), (73, 74), (83, 84), (93, 94)]
invalidTransitions += [(24, 23), (34, 33), (44, 43), (54, 53), (64, 63), (74, 73), (84, 83), (94, 93)]
invalidTransitions += [(36, 46), (37, 47), (38, 48), (39, 49)]
invalidTransitions += [(46, 36), (47, 37), (48, 38), (49, 39)]
invalidTransitions += [(76, 86), (77, 87), (86, 76), (87, 77)]
invalidTransitions += [(45, 46), (55, 56), (65, 66), (75, 76)]
invalidTransitions += [(46, 45), (56, 55), (66, 65), (76, 75)]
labyrinth = make_gridworld(10, 10, terminals=[0], rewards=np.array([[0, 1]]), invalid_transitions=invalidTransitions, goals=[0])
labyrinth['starting_states'] = np.array([4])
pickle.dump(labyrinth, open('environments/labyrinth.pkl', 'wb'))
