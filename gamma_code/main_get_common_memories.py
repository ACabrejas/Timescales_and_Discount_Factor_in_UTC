from Vissim_env_class import environment
from MasterDQN_Agent import MasterDQN_Agent

# General Libraries
import numpy as np
import pylab as plt
import os
import csv
import pandas as pd
import json

model_name = 'Single_Cross_Triple'
# vissim_working_directory =  'C:\\Users\\Rzhang\\Desktop\\MLforFlowOptimisationOrigine\\Vissim\\'
vissim_working_directory = 'C:\\Users\\acabrejasegea\\Desktop\\15_Timescales_utc\\gamma_code'

sim_length = 3601
timesteps_per_second = 1
learning_iterations = 10
actions_set = "all_actions"

agent_type = "DuelingDDQN"
Session_ID = "SCT_8act_DuelingDDQN_acf_memorytest"

## DQN Hyperaramenters
episodes = 5
copy_weights_frequency = 20 # On a successfull run I copied the weight every 50

PER_activated = True
memory_size = 1000
batch_size = 128

gamma = 0.95
alpha = 0.005

# Exploration Schedule ("linear" or "geometric")
exploration_schedule = "geometric"
epsilon_start = 1
epsilon_end   = 0.001

Random_Seed = 100

# all controller actions
Single_Cross_Triple_dictionary8 = \
    { \
        # Controller Number 0
        'junctions': {0: {'default_actions': {0: [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                              1: [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                                              2: [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
                                              3: [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                                              4: [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                                              5: [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
                                              6: [0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0],
                                              7: [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1]},

                          'all_actions': {0: [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                          1: [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                                          2: [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
                                          3: [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                                          4: [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                                          5: [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
                                          6: [0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0],
                                          7: [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1]},

                          'link': [1, 3, 5, 7],
                          'lane': ['1-1', '1-2', '1-3', '3-1', '3-2', '3-3', '5-1', '5-2', '5-3', '7-1', '7-2', '7-3'],

                          'controled_by_com': True,
                          'agent_type': agent_type,
                          'green_time': 6,
                          'redamber_time': 0,
                          'amber_time': 3,
                          'red_time': 0,
                          'state_size': [13],
                          'state_type': 'QueuesSig',
                          'reward_type': 'Queues',
                          'queues_counter_ID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
                          }
                      },
        'demand': {'default': [400, 400, 400, 400],

                   0: [300, 300, 300, 300],
                   1: [600, 600, 600, 600],
                   2: [1350, 750, 1350, 750],
                   3: [1500, 750, 1500, 750],
                   4: [1050, 750, 1050, 750],
                   5: [750, 1050, 750, 1050],
                   6: [750, 1500, 750, 1500],
                   7: [750, 1350, 750, 1350],
                   8: [600, 600, 600, 600],
                   9: [300, 300, 300, 300]
                   }
    }


def choose_schedule(exploration_schedule, espilon_start, epsilon_end, episodes):
    if exploration_schedule == "linear":
        epsilon_decay = 1.2 * (epsilon_end - epsilon_start) / (episodes - 1)
        epsilon_sequence = [1 + epsilon_decay * entry for entry in range(episodes + 1)]
        epsilon_sequence = [0 if entry < 0 else entry for entry in epsilon_sequence]
    elif exploration_schedule == "geometric":
        epsilon_decay = np.power(epsilon_end / epsilon_start, 1. / (episodes - 1))  # Geometric decay
        epsilon_sequence = [epsilon_start * epsilon_decay ** entry for entry in range(episodes + 1)]
    elif exploration_schedule == "entropy":
        pass
    else:
        print("ERROR: Unrecognized choice of exploration schedule.")

    # Plotting exploration schedule
    plt.figure(figsize=(8, 4.5))
    x_series = np.array(range(1, episodes + 1))
    y_series = epsilon_sequence[0:episodes]
    plt.plot(x_series, y_series, '-b')
    plt.xlabel('Episodes')
    plt.ylabel('Ratio of random exploration')
    plt.title('Exploration schedule')
    plt.show()
    return (epsilon_sequence)


epsilon_sequence = choose_schedule(exploration_schedule, epsilon_start, epsilon_end, episodes)

Single_Cross_Triple8_MultiDQN_Agents = MasterDQN_Agent(model_name, vissim_working_directory,\
                                                       sim_length, Single_Cross_Triple_dictionary8,\
                                                       actions_set, gamma, alpha, agent_type,\
                                                       memory_size, PER_activated, batch_size,\
                                                       learning_iterations, copy_weights_frequency,\
                                                       epsilon_sequence,Random_Seed,\
                                                       timesteps_per_second, Session_ID,\
                                                       verbose = True)

Single_Cross_Triple8_MultiDQN_Agents.prepopulate_memory()

