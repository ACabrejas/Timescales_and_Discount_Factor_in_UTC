import numpy as np
import random
from General_agent import RLAgent



###
######################################################################################
## Deep Q Learning Agent (Use DoubleDQN flag to swap to DDQN)
######################################################################################

class MOAgent():
    def __init__(self, ID):
        # Agent Junction ID and Controller ID
        self.signal_id = ID

        # Metrics for the testing
        self.queues_over_time = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
        self.delay = [0]

    def choose_action(self, state):
        '''
            Input : State as an array.
            Output: Action as an integer.
        '''

        self.queues_over_time.append(state)
        action_values = [state[0][0]+state[0][1]+state[0][2],
                         state[0][3]+state[0][4]+state[0][5],
                         state[0][6]+state[0][7]+state[0][8],
                         state[0][9]+state[0][10]+state[0][11],
                         state[0][0]+state[0][6],
                         state[0][3]+state[0][9],
                         state[0][1]+state[0][2]+state[0][7]+state[0][8],
                         state[0][4]+state[0][5]+state[0][10]+state[0][11]]

        action = np.argmax(action_values)
        return action



