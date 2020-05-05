from MOAgents import MOAgent
from Vissim_env_class import environment
import os
import pickle
import numpy as np
from time import time


class MasterMO_Agent():
    """
    A Master class agent containing the other agents.

    """

    def __init__(self, model_name, vissim_working_directory, sim_length, Model_dictionnary, actions_set, \
                 Random_Seed, timesteps_per_second, Session_ID, verbose=True):

        # Model information
        self.Model_dictionnary = Model_dictionnary
        self.model_name = model_name
        self.sim_length = sim_length
        self.actions_set = actions_set
        self.vissim_working_directory = vissim_working_directory
        self.timesteps_per_second = timesteps_per_second

        # Simulation Parameters
        self.Random_Seed = Random_Seed

        # For saving put here all relevent information and saving parameters
        self.Session_ID = Session_ID

        # Spawn one individual agent per junction
        self.Agents = {}

        current_Agent = 0
        for idx, info in Model_dictionnary['junctions'].items():
            if info['controled_by_com']:
                print("INTERSECTION " + str(idx) + ": SETTING UP AGENT")
                self.Agents[current_Agent] = MOAgent(idx)
            current_Agent += 1

    def get_data(self):
        """
        Function to train the agents
        input the number of episode of training

        """
        self.env = None
        self.env = environment(self.model_name, self.vissim_working_directory, self.sim_length, self.Model_dictionnary,
                               self.actions_set, \
                               self.Random_Seed, timesteps_per_second=self.timesteps_per_second, mode='training',
                               delete_results=True, verbose=True)

        # Get initial State
        start_state = self.env.get_state()
        print("start")
        # Episodic training loop

        # Create dictionary for chosen actions for each agent and fill it
        actions = {}
        for idx, s in start_state.items():
            actions[idx] = self.Agents[idx].choose_action(s)
        # Simulation Loop, Run until end of simulation

        t = 0
        while (self.sim_length-3) > self.env.global_counter:
            #self.Agents[idx].delay.append(self.env.SCUs[0].calculate_delay())
            SARSDs = self.env.step_to_next_action(actions)
            actions = dict()
            for idx, sarsd in SARSDs.items():
                s, a, r, ns, d = sarsd
                # in order to find the next action you need to evaluate the "next_state" because it is the current state of the simulator
                actions[idx] = int(self.Agents[idx].choose_action(ns))
            t += 1
        self.env = None

    def demo(self):

        self.env = None
        self.env = environment(self.model_name, self.vissim_working_directory, self.sim_length, self.Model_dictionnary,
                               self.actions_set, \
                               Random_Seed=self.Random_Seed, timesteps_per_second=self.timesteps_per_second,
                               mode='demo', delete_results=True, verbose=True)

        start_state = self.env.get_state()

        actions = {}
        for idx, s in start_state.items():
            actions[idx] = self.Agents[idx].choose_action(s)
        # Simulation Loop, Run until end of simulation

        t = 0
        while (self.sim_length - 3) > self.env.global_counter:
            # self.Agents[idx].delay.append(self.env.SCUs[0].calculate_delay())
            SARSDs = self.env.step_to_next_action(actions)
            actions = dict()
            for idx, sarsd in SARSDs.items():
                s, a, r, ns, d = sarsd
                # in order to find the next action you need to evaluate the "next_state" because it is the current state of the simulator
                actions[idx] = int(self.Agents[idx].choose_action(ns))
            t += 1
        self.env = None