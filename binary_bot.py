# -*- coding: utf-8 -*-
"""
@author: Eric Born
Developed a bot that plays at the Protoss race
Choses a random difficulty between 0-9 then launches SC2
and plays against the built-in AI
Keeps track 12 attributes of the games progress and writes the results
out to a numpy array file
Also appends the outcome of the match to a csv file. 
-1 for loss, 0 for tie, 1 for win
"""
# general libraries
import numpy as np
import pandas as pd
import random
#import math
import time
import csv
import os

# NN model specific
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# pysc2 libraries
from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features
from pysc2.env import sc2_env

# sc2 libraries
import sc2
from sc2 import run_game, maps, Race, Difficulty, Result
from sc2.player import Bot, Computer
from sc2.constants import NEXUS, PROBE, PYLON, ASSIMILATOR, \
 CYBERNETICSCORE, GATEWAY, ROBOTICSBAY, ROBOTICSFACILITY, STARGATE, \
 ZEALOT, STALKER, ADEPT, IMMORTAL, VOIDRAY, COLOSSUS

# Q learning system found here:
# https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow
# class QLearningTable:
#     def __init__(self, actions, learning_rate=0.01,
#                  reward_decay=0.9, e_greedy=0.9):
#         self.actions = actions  # a list
#         self.lr = learning_rate
#         self.gamma = reward_decay
#         self.epsilon = e_greedy
#         self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

#     def choose_action(self, observation):
#         self.check_state_exist(observation)

#         if np.random.uniform() < self.epsilon:
#             # choose best action
#             state_action = self.q_table.ix[observation, :]

#             # some actions have the same value
#             state_action = state_action.reindex(np.random.permutation(
#                                                 state_action.index))

#             action = state_action.idxmax()
#         else:
#             # choose random action
#             action = np.random.choice(self.actions)

#         return action

#     def learn(self, s, a, r, s_):
#         self.check_state_exist(s_)
#         self.check_state_exist(s)

#         q_predict = self.q_table.ix[s, a]
#         q_target = r + self.gamma * self.q_table.ix[s_, :].max()

#         # update
#         self.q_table.ix[s, a] += self.lr * (q_target - q_predict)

#     def check_state_exist(self, state):
#         if state not in self.q_table.index:
#             # append new state to q table
#             self.q_table = self.q_table.append(pd.Series([0] *
#                                                len(self.actions),
#                                                index=self.q_table.columns,
#                                                name=state))

#choice = random.randint(0, 8)

# TODO
#### USE DESCRETE ACTIONS AS INPUTS TO THE RL algorithm 
# Dueling-DDQN [7, 38, 39] and PPO [40]), 
# together with a distributed rollout infrastructure.

# TODO
# TENCENT IS USING A REWARD SYSTEM AFTER EACH STEP, 
# SEE IF YOU CAN FIND WHAT IT IS.
# INFORMATION SEEMS CONFLICTING, ONE SAYS AFTER EACH STEP THEN REFER TO A 
# SECTION THAT ONLY COVERS THE END OF GAME REWARD.
# POSSIBLY CREATE A SYSTEM THAT PENALIZES CHOOSING AN OPTION THAT ISNT 
# CURRENTLY AVAILABLE SO THAT THE BOT IS LESS LIKELY TO CHOSE IT ON THE NEXT 
# STEP. AFTER IT COMPLETES A STEP THE PENALTY GOES AWAY.

# units unlocked by the following buildings
# gateway - ZEALOT
# cyber core - STALKER, ADEPT
# robo facility - IMMORTAL
# stargate - VOIDRAY
# robo bay - COLOSSUS
unit_list = [
    None,
    'ZEALOT',
    'STALKER',
    'ADEPT',
    'IMMORTAL',
    'VOIDRAY',
    'COLOSSUS',
]

# building indicators, used to check if units can be created
# Flipped to a 1 if they exist
GATEWAY_IND = 0
CYBERCORE_IND = 0
ROBOFACILITY_IND = 0
STARGATE_IND = 0
ROBOBAY_IND = 0

unit_choice = ''

# training_data format and locations
# supply/building levels stored in the supply_data array
# [0]total minerals, [1]total gas,
# [2]supply_cap, [3]supply_army, [4]supply_workers, [5]NEXUS, [6]PYLON, 
# [7]ASSIMILATOR, [8]GATEWAY, [9]CYBERCORE, [10]ROBOFAC, [11]STARGATE, 
# [12]ROBOBAY, [13]killed_structures, [14]killed_units

# actions stored in action_data array
# [15]attack, [16]assimilators, [17]offensive_force, [18]pylons, [19]workers, 
# [20]distribute,  [21]]nothing, [22]expand, [23]buildings

# troops stored in the troop_data array
# [24]ZEALOT, [25]STALKER, [26]ADEPT, [27]IMMORTAL, [28]VOIDRAY, [29]COLOSSUS

# outcome info stored in the outcome_data array
# [30]difficulty, [31]outcome

# Creates a random number between 0-9
# this is used in the main() to set the difficulty of the game
diff = 2 #random.randrange(0,10)

diff_dict = {
    0:'VeryEasy', 
    1:'Easy', 
    2:'Medium',
    3:'MediumHard', 
    4:'Hard', 
    5:'Harder',
    6:'VeryHard', 
    7:'CheatMoney', 
    8:'CheatVision', 
    9:'CheatInsane'
}

# Isnt working because you cant pass the difficulty as a string
# diff_dict = {
#     0:'Difficulty.VeryEasy', 
#     1:'Difficulty.Easy', 
#     2'Difficulty.Medium,'
#     3'Difficulty.MediumHard', 
#     4'Difficulty.Hard', 
#     5'Difficulty.Harder',
#     6'Difficulty.VeryHard', 
#     7'Difficulty.CheatMoney', 
#     8'Difficulty.CheatVision', 
#     9'Difficulty.CheatInsane'
# }

# maps the functions from the pysc2 actions file
FUNCTIONS = actions.FUNCTIONS

# removes scientific notation from np prints, prints numbers as floats
np.set_printoptions(suppress=True)

# Bots current issues:
# Troops need to move out to protect expanded bases
# sometimes creates multiple nexuses right next to each other
# wont target troops repairing buildings
class BinaryBot(sc2.BotAI):
    def __init__(self, use_model=False):
        self.MAX_WORKERS = 50
        
        # path to record csv
        self.csv_path = 'C:/Users/TomBrody/Desktop/School/767 ML/SC Bot/Q-learn/record.csv'
        self.text_path = 'C:/Users/TomBrody/Desktop/School/767 ML/SC Bot/Q-learn/record.txt'

        # Used to slow down the bots actions
        self.do_something_after = 0
        self.delay_time = 0
        self.delay = 25

        # used to scale the supply data before the model makes a prediction
        self.x_scaler = MinMaxScaler()

        # set the path to the model
        self.model_path = 'C:/Users/TomBrody/Desktop/School/767 ML/SC Bot/NN/model/CuDNNLSTM-1571072630.h5'

        # used to load a trained model to choose actions instead of random
        self.model = load_model(self.model_path)
        
        # stores actions taken
        self.action_data = np.zeros(9)

        # store data relating to current troop/building numbers
        # updated every 5th iteration
        self.supply_data = np.zeros(15)

        # stores data related to which troops were built
        self.troop_data = np.zeros(6)

        # stores supply_data, action_data, troop_data and outcome
        self.training_data = []
        
        # Store the difficulty setting in the array that is used as output data
        self.outcome_data = np.zeros(2)
        self.outcome_data[0] = diff

        # Setup actions dictionary
        self.actions_dict = {
            0: self.attack,
            1: self.build_assimilators,
            2: self.build_offensive_force,
            3: self.build_pylons,
            4: self.build_workers,
            5: self.d_distribute_workers,
            6: self.do_nothing,
            7: self.expand,
            8: self.offensive_force_buildings
        }
        
        self.action_count = {
            0: 1,
            1: 1,
            2: 1,
            3: 1,
            4: 1,
            5: 1,
            6: 1,
            7: 1,
            8: 1
        }
        
        
       
    # self.qlearn = QLearningTable(actions=list(range(len(smart_actions))))
    
    def reset_counts(self):
        #global action_count
        self.action_count = {
            0: 1,
            1: 1,
            2: 1,
            3: 1,
            4: 1,
            5: 1,
            6: 1,
            7: 1,
            8: 1
        }
    # creates a dictionary to store counts of actions
    #reset_counts()
    
    # Create a function to write the result to a text file
    def write_txt(self, result, diff):
        with open(self.text_path,'a') as textfile:
            outcome = str(result) + ',' + str(diff) + '\n'
            textfile.write(outcome)
            textfile.close()

    # Create a function to write the result to a csv
#    def write_csv(self, game_result, difficulty):
#        with open(self.csv_path,'a', newline='') as csvfile:
#            writer = csv.writer(csvfile)
#            result = [str(game_result), str(difficulty)]
#            writer.writerow(result)
                
#    def write_csv(self, game_result):
#        with open('record.csv','a', newline='') as csvfile:
#                writer = csv.writer(csvfile)
#                writer.writerow(game_result)

    def on_end(self, game_result):
        result = str(game_result)
        
        # Defeat
        if result == 'Result.Defeat':
            self.outcome_data[1] = -1
            self.training_data.append([
                self.state.score.collected_minerals,
                self.state.score.collected_vespene,
                self.supply_cap, self.supply_army,
                self.supply_workers, self.units(NEXUS).amount,
                self.units(PYLON).amount, self.units(ASSIMILATOR).amount,
                self.units(GATEWAY).amount, self.units(CYBERNETICSCORE).amount,
                self.units(ROBOTICSFACILITY).amount, self.units(STARGATE).amount,
                self.units(ROBOTICSBAY).amount,
                self.state.score.killed_value_structures,
                self.state.score.killed_value_units
            ])
            self.training_data[-1].extend(self.action_data)
            self.training_data[-1].extend(self.troop_data)
            self.training_data[-1].extend(self.outcome_data)
            print(str(-1), diff_dict[diff])

            self.write_txt(-1, diff_dict[diff])
            #self.write_csv(str(-1), diff_dict[diff])
            #self.write_csv([-1])
            
            np.save(r"C:/botdata/{}.npy".format(str(int(time.time()))),
                    np.array(self.training_data))
        
        # Win
        elif result == 'Result.Victory':
            self.outcome_data[1] = 1
            self.training_data.append([
                self.state.score.collected_minerals,
                self.state.score.collected_vespene,
                self.supply_cap, self.supply_army,
                self.supply_workers, self.units(NEXUS).amount,
                self.units(PYLON).amount, self.units(ASSIMILATOR).amount,
                self.units(GATEWAY).amount, self.units(CYBERNETICSCORE).amount,
                self.units(ROBOTICSFACILITY).amount, self.units(STARGATE).amount,
                self.units(ROBOTICSBAY).amount,
                self.state.score.killed_value_structures,
                self.state.score.killed_value_units
            ])
            self.training_data[-1].extend(self.action_data)
            self.training_data[-1].extend(self.troop_data)
            self.training_data[-1].extend(self.outcome_data)
            print(str(1), diff_dict[diff])
            
            #self.write_csv(1, diff_dict[diff])
            self.write_txt(1, diff_dict[diff])
            #self.write_csv([1])
            
            np.save(r"C:/botdata/{}.npy".format(str(int(time.time()))),
                    np.array(self.training_data))
        
        # Tie
        else:
            self.outcome_data[1] = 0
            self.training_data.append([
                self.state.score.collected_minerals,
                self.state.score.collected_vespene,
                self.supply_cap, self.supply_army,
                self.supply_workers, self.units(NEXUS).amount,
                self.units(PYLON).amount, self.units(ASSIMILATOR).amount,
                self.units(GATEWAY).amount, self.units(CYBERNETICSCORE).amount,
                self.units(ROBOTICSFACILITY).amount, self.units(STARGATE).amount,
                self.units(ROBOTICSBAY).amount,
                self.state.score.killed_value_structures,
                self.state.score.killed_value_units
            ])
            self.training_data[-1].extend(self.action_data)
            self.training_data[-1].extend(self.troop_data)
            self.training_data[-1].extend(self.outcome_data)  
            print(str(0), diff_dict[diff])
            
            #self.write_csv(0, diff_dict[diff])
            self.write_txt(0, diff_dict[diff])
            #self.write_csv([0])

            np.save(r"C:/botdata/{}.npy".format(str(int(time.time()))),
                    np.array(self.training_data))

    # This is the function that steps forward
    # and is called through each frame of the game
    async def on_step(self, iteration):
        # 22.4 per second on faster game speed
        self.time_loop = (self.state.game_loop/22.4) / 60

        # functions to select next action and send idle workers to a task
        await self.smart_action()
        await self.back_to_work()
        await self.distribute_workers()
        await self.first_pylon()

        # send starting chat message
        if iteration == 0:
            await self.chat_send("(glhf)")
    
    #!!! NOT FIRING!!!
    async def first_pylon(self):
        #print('first pylon')
        if not self.already_pending(PYLON) and self.units(PYLON).amount == 0:
            nexuses = self.units(NEXUS).ready
            if nexuses.exists:
                if self.can_afford(PYLON):
                    await self.build(PYLON, near=
                                     self.units(NEXUS).first.position.towards(
                                             self.game_info.map_center, 5))

#    async def build_pylons(self):
#        #print('build_pylons')
#        if self.supply_cap != 200 and self.supply_left < 10: 
#            #and not self.already_pending(PYLON): #dont care, build another
#            nexuses = self.units(NEXUS).ready
#            if nexuses.exists:
#                if self.can_afford(PYLON):
#                    await self.build(PYLON, near=
#                                     self.units(NEXUS).first.position.towards(
#                                             self.game_info.map_center, 5))

            
    # attempt to fix workers starting the warp in of a building
    # and not going back to work until its finished.
    # checks for idle workers then calls a distribute_workers
    # to send them back to work.
    # Does not work on workers who create assimilators since they're
    # being assigned to get gas upon starting the build
    async def back_to_work(self):
        #if self.idle_worker_count > 0:
        self.distribute_workers()
        #print('distribute')

    def find_target(self, state):
        return self.enemy_start_locations[0]
                
    # seems to send troops to attack but if they start to become attacked
    # they dont fight back, just running to some location
    # may be due to attack a building some other troop saw,
    # not a general attack command toward the buildings
#    def find_target(self, state):
#            if len(self.known_enemy_units) > 0:
#                return random.choice(self.known_enemy_units)
#            elif len(self.known_enemy_structures) > 0:
#                return random.choice(self.known_enemy_structures)
#            else:
#                return self.enemy_start_locations[0]
#                
    # Action 0 - Attack
    async def attack(self):
        # print('attack')
        attack_amount = random.randrange(5, 10)
        if self.units.of_type([ZEALOT, STALKER, ADEPT, IMMORTAL, VOIDRAY, 
                               COLOSSUS]).amount > attack_amount:
            for s in self.units.of_type([ZEALOT, STALKER, ADEPT, IMMORTAL, 
                                         VOIDRAY, COLOSSUS]).idle:
                await self.do(s.attack(self.find_target(self.state)))

    # Action 1 - build assimilators
    # TODO
    # need to add check to move probes onto gas at this same step
    async def build_assimilators(self):
        # print('build_assimilators')
        if self.supply_cap > 16:
            for nexus in self.units(NEXUS).ready:
                vaspenes = self.state.vespene_geyser.closer_than(15.0, nexus)
                for vaspene in vaspenes:
                    if not self.can_afford(ASSIMILATOR):
                        break
                    worker = self.select_build_worker(vaspene.position)
                    if worker is None:
                        break
                    if not self.units(ASSIMILATOR).closer_than(
                                                    1.0, vaspene).exists:
                        await self.do(worker.build(ASSIMILATOR, vaspene))
   
    # Action 2 - build offensive force
    async def build_offensive_force(self):
        # print('build_offensive_force')
        # updates variables that indicate if a building exists
        # used to check if a unit can be built
        
        # troop data contains the type of troop
        # selected for creation
        # re-initialized each time build_offensive_force
        # is called to prevent object reference issues
        # in the training_data list
        self.troop_data = np.zeros(6)

        if self.units(GATEWAY).ready.exists:
            #print('gateway exists')
            GATEWAY_IND = 1
        else:
            GATEWAY_IND = 0

        if self.units(CYBERNETICSCORE).ready.exists:
            #print('cyber exists')
            CYBERCORE_IND = 1
        else:
            CYBERCORE_IND = 0

        if self.units(ROBOTICSFACILITY).ready.exists:
            #print('robo-fac exists')
            ROBOFACILITY_IND = 1
        else:
            ROBOFACILITY_IND = 0

        if self.units(STARGATE).ready.exists:
            #print('stargate exists')
            STARGATE_IND = 1
        else:
            STARGATE_IND = 0

        if self.units(ROBOTICSBAY).ready.exists:
            #print('robo-bay exists')
            ROBOBAY_IND = 1
        else:
            ROBOBAY_IND = 0

        # random choice of what unit to build
        # limited by the buildings that unlock the unit being built
        if ROBOBAY_IND == 1 and ROBOFACILITY_IND == 1:
            #print('random 1-6')
            unit_choice = unit_list[random.randint(1, 6)]

        elif ROBOFACILITY_IND == 1 and STARGATE_IND == 1:
            #print('random 1-5')
            unit_choice = unit_list[random.randint(1, 5)]

        elif ROBOFACILITY_IND == 1 and STARGATE_IND == 0:
            #print('random 1-4')
            unit_choice = unit_list[random.randint(1, 4)]

        elif CYBERCORE_IND == 1:
            #print('random 1-3')
            unit_choice = unit_list[random.randint(1, 3)]

        elif GATEWAY_IND == 1:
            #print('zealot')
            unit_choice = unit_list[1]

        else:
            #print('none')
            unit_choice = unit_list[0]

        
        # TODO
        # currently only queues one unit at a time using gw.train
        # hacky method is just to call it multiple times per troop
        # a better method should be found
        if unit_choice == 'ZEALOT' and self.can_afford(ZEALOT) and \
        self.supply_left >= 2:
            for gw in self.units(GATEWAY).ready.idle:
                await self.do(gw.train(ZEALOT))
                self.troop_data[0] = 1

        elif unit_choice == 'STALKER' and self.can_afford(STALKER) and \
        self.supply_left >= 2:
            for gw in self.units(GATEWAY).ready.idle:
                await self.do(gw.train(STALKER))
                await self.do(gw.train(STALKER))
                await self.do(gw.train(STALKER))
                self.troop_data[1] = 1

        elif unit_choice == 'ADEPT' and self.can_afford(ADEPT) and \
        self.supply_left >= 2:
            for gw in self.units(GATEWAY).ready.idle:
                await self.do(gw.train(ADEPT))
                await self.do(gw.train(ADEPT))
                await self.do(gw.train(ADEPT))
                self.troop_data[2] = 1

        elif unit_choice == 'IMMORTAL' and self.can_afford(IMMORTAL) and \
        self.supply_left >= 4:
            for gw in self.units(ROBOTICSFACILITY).ready.idle:
                await self.do(gw.train(IMMORTAL))
                await self.do(gw.train(IMMORTAL))
                await self.do(gw.train(IMMORTAL))
                self.troop_data[3] = 1

        elif unit_choice == 'VOIDRAY' and self.can_afford(VOIDRAY) and \
        self.supply_left >= 4:
            for gw in self.units(STARGATE).ready.idle:
                await self.do(gw.train(VOIDRAY))
                await self.do(gw.train(VOIDRAY))
                await self.do(gw.train(VOIDRAY))
                self.troop_data[4] = 1

        elif unit_choice == 'COLOSSUS' and self.can_afford(COLOSSUS) and \
        self.supply_left >= 6:
            for gw in self.units(ROBOTICSFACILITY).ready.idle:
                await self.do(gw.train(COLOSSUS))
                await self.do(gw.train(COLOSSUS))
                await self.do(gw.train(COLOSSUS))
                self.troop_data[5] = 1
    
    # action 3
    async def build_pylons(self):
        #print('build_pylons')
        if self.supply_cap != 200 and self.supply_left < 10: 
            #and not self.already_pending(PYLON): #dont care, build another
            nexuses = self.units(NEXUS).ready
            if nexuses.exists:
                if self.can_afford(PYLON):
                    await self.build(PYLON, near=
                                     self.units(NEXUS).first.position.towards(
                                             self.game_info.map_center, 5))

    # action 4
    # builds 16 workers per nexus up to a maximum of 50
    async def build_workers(self):
        #print('build_workers')
        if (self.units(NEXUS).amount * 16) > self.units(PROBE).amount and \
            self.units(PROBE).amount < self.MAX_WORKERS:
            for nexus in self.units(NEXUS).ready.idle:
                if self.can_afford(PROBE):
                    await self.do(nexus.train(PROBE))

    # action 5
    async def d_distribute_workers(self):
        if self.idle_worker_count > 0:
            print('workers')
        #self.distribute_workers()


    # action 6
    async def do_nothing(self):
        #print('do_nothing')
        wait = random.randrange(10, 30)/100
        self.do_something_after = self.time_loop + wait

    # action 7
    # Added not already_pending trying to prevent multiple
    # being built right next to each other
    async def expand(self):
        #print('expand')
        if self.can_afford(NEXUS) and \
            not self.already_pending(NEXUS):
            await self.expand_now()

    # action 8
    async def offensive_force_buildings(self):
        #print('offensive_force_buildings')
        # Checks for a pylon as an indicator of where to build
        # small area around pylon is needed to place another building
        if self.units(PYLON).ready.exists:
            pylon = self.units(PYLON).ready.random

            # Gateway required first
            if self.can_afford(GATEWAY) and \
                not self.already_pending(GATEWAY):
                #and self.units(GATEWAY).amount <= 2:
                await self.build(GATEWAY, near=pylon)
            
            if self.units(GATEWAY).ready.exists and \
               self.units(CYBERNETICSCORE).amount < 1: # Added to limit to 1
                if self.can_afford(CYBERNETICSCORE) and not \
                   self.already_pending(CYBERNETICSCORE):
                    await self.build(CYBERNETICSCORE, near=pylon)

            if self.units(CYBERNETICSCORE).ready.exists:
                if self.can_afford(ROBOTICSFACILITY) and not \
                    self.already_pending(ROBOTICSFACILITY):
                    await self.build(ROBOTICSFACILITY, near=pylon)
            
            if self.units(CYBERNETICSCORE).ready.exists:
                if self.can_afford(STARGATE) and not \
                    self.already_pending(STARGATE):
                    await self.build(STARGATE, near=pylon)

            if self.units(ROBOTICSFACILITY).ready.exists:
                if self.can_afford(ROBOTICSBAY) and not \
                    self.already_pending(ROBOTICSBAY):
                    await self.build(ROBOTICSBAY, near=pylon)
        else:
            nexuses = self.units(NEXUS).ready
            if nexuses.exists and not self.already_pending(PYLON):
                if self.can_afford(PYLON):
                    await self.build(PYLON, near=
                                     self.units(NEXUS).first.position.towards(
                                             self.game_info.map_center, 5))

    # self.state.game_loop moves at 22.4 per second on faster game speed
    # Hacky attempt at throttling the bots actions using the time from
    # state.game_loop with an added delay of 25. Should represent
    # about 1 move a second. 60-100 APM is average, 200+ is pro
    # https://github.com/deepmind/pysc2/blob/master/docs/environment.md#apm-calculation
    async def smart_action(self):
#        if self.state.game_loop > self.delay_time and \
#            self.time_loop > self.do_something_after:
        if self.state.game_loop > 0 and \
            self.time_loop > 0:

            # choses random number which represents the action
            # being carried out on the next step
            # re-initialized each time smart_action
            # is called to prevent object reference issues
            # in the training_data list
            self.action_data = np.zeros(9)

            # records all of the current stats about the match
            # this data is then fed into the model and it makes
            # a prediction on which move to use next
            self.supply_data[0] = self.state.score.collected_minerals
            self.supply_data[1] = self.state.score.collected_vespene
            self.supply_data[2] = self.supply_cap
            self.supply_data[3] = self.supply_army
            self.supply_data[4] = self.supply_workers
            self.supply_data[5] = self.units(NEXUS).amount
            self.supply_data[6] = self.units(PYLON).amount
            self.supply_data[7] = self.units(ASSIMILATOR).amount
            self.supply_data[8] = self.units(GATEWAY).amount
            self.supply_data[9] = self.units(CYBERNETICSCORE).amount
            self.supply_data[10] = self.units(ROBOTICSFACILITY).amount
            self.supply_data[11] = self.units(ROBOTICSBAY).amount
            self.supply_data[12] = self.state.score.killed_value_structures
            self.supply_data[13] = self.state.score.killed_value_units

            #print(self.supply_data)

            # shape data before scaling
            x_data = self.supply_data.reshape(-1, 1)
            #print(x_data)

            # min/max scale the data
            x_scaled = self.x_scaler.fit_transform(x_data)
            #print(x_scaled)

            # shape again before prediction
            x_shaped_scaled = x_scaled.reshape(1, -1)
            #print(x_shaped_scaled)
            
            # since the predictions are just percentage chances between each of the
            # actions we take the maximum percentage as the action choice.
            # since x_test_scaled is a single row we have to manipulate
            # the shape of the data to something the model is used to seeing.
            model_choice = np.argmax(self.model.predict(np.array([x_shaped_scaled,])))

            # fix for model continuously choosing the same action
            # if the action is chosen 5 times in a row and two other
            # random actions have both only been chosen once
            # a random action is selected and the counts are reset
            if self.action_count.get(model_choice) > 4 and \
                self.action_count[random.randrange(0, 9)] == 1 and \
                self.action_count[random.randrange(0, 9)] == 1:
                    action_choice = random.randrange(0, 9)
                    print('random choice', action_choice)
                    self.reset_counts()
            else:
                action_choice = model_choice
                self.action_count[action_choice] += 1
                print('model choice', action_choice)
            
            #action_choice = model_choice
            print('model choice', action_choice)
            #action_choice = random.randrange(0, 9)
            #print(self.actions_dict[action_choice])

            # appends all supply data to the training_data list
            # then extends that list with the action, troop and outcome data
            # in the last index with -1
            # on the next loop it appends a new list to the original list 
            # and repeats the process
            if action_choice == 2:
                self.training_data.append([
                    self.state.score.collected_minerals,
                    self.state.score.collected_vespene,
                    self.supply_cap, self.supply_army,
                    self.supply_workers, self.units(NEXUS).amount,
                    self.units(PYLON).amount, self.units(ASSIMILATOR).amount,
                    self.units(GATEWAY).amount, self.units(CYBERNETICSCORE).amount,
                    self.units(ROBOTICSFACILITY).amount, self.units(STARGATE).amount,
                    self.units(ROBOTICSBAY).amount,
                    self.state.score.killed_value_structures,
                    self.state.score.killed_value_units
                ])
                self.training_data[-1].extend(self.action_data)
                self.training_data[-1].extend(self.troop_data)
                self.training_data[-1].extend(self.outcome_data)
            else:
                self.training_data.append([
                    self.state.score.collected_minerals,
                    self.state.score.collected_vespene,
                    self.supply_cap, self.supply_army,
                    self.supply_workers, self.units(NEXUS).amount,
                    self.units(PYLON).amount, self.units(ASSIMILATOR).amount,
                    self.units(GATEWAY).amount, self.units(CYBERNETICSCORE).amount,
                    self.units(ROBOTICSFACILITY).amount, self.units(STARGATE).amount,
                    self.units(ROBOTICSBAY).amount,
                    self.state.score.killed_value_structures,
                    self.state.score.killed_value_units
                ])
                self.training_data[-1].extend(self.action_data)
                self.training_data[-1].extend([0,0,0,0,0,0])
                self.training_data[-1].extend(self.outcome_data)
            
            # print various parts from training_data
            #print(self.training_data[-1][0:2])
            
            try:
                await self.actions_dict[action_choice]()
            except Exception as e:
                print(str(e))
            # Only gets appended
            # self.training_data.append([self.supply_data, self.action_data, 
            #                            self.troop_data, self.outcome_data])
            self.delay_time = self.state.game_loop + self.delay
            
# Fixed difficulty
def main():
    run_game(maps.get("AbyssalReefLE"), [
        Bot(Race.Protoss, BinaryBot()),
        Computer(Race.Terran, Difficulty.Medium)
        ], realtime=True)

if __name__ == '__main__':
    main()

# Random difficulty
# def main():
#    if diff == 0:
#        run_game(maps.get("AbyssalReefLE"), [
#            Bot(Race.Protoss, BinaryBot()),
#            Computer(Race.Terran, Difficulty.VeryEasy)
#            ], realtime=False)

#    if diff == 1:
#        run_game(maps.get("AbyssalReefLE"), [
#            Bot(Race.Protoss, BinaryBot()),
#            Computer(Race.Terran, Difficulty.Easy)
#            ], realtime=False)
   
#    if diff == 2:
#        run_game(maps.get("AbyssalReefLE"), [
#            Bot(Race.Protoss, BinaryBot()),
#            Computer(Race.Terran, Difficulty.Medium)
#            ], realtime=False)

#    if diff == 3:
#        run_game(maps.get("AbyssalReefLE"), [
#            Bot(Race.Protoss, BinaryBot()),
#            Computer(Race.Terran, Difficulty.MediumHard)
#            ], realtime=False)
   
#    if diff == 4:
#        run_game(maps.get("AbyssalReefLE"), [
#            Bot(Race.Protoss, BinaryBot()),
#            Computer(Race.Terran, Difficulty.Hard)
#            ], realtime=False)

#    if diff == 5:
#        run_game(maps.get("AbyssalReefLE"), [
#            Bot(Race.Protoss, BinaryBot()),
#            Computer(Race.Terran, Difficulty.Harder)
#            ], realtime=False)
   
#    if diff == 6:
#        run_game(maps.get("AbyssalReefLE"), [
#            Bot(Race.Protoss, BinaryBot()),
#            Computer(Race.Terran, Difficulty.VeryHard)
#            ], realtime=False)

#    if diff == 7:
#        run_game(maps.get("AbyssalReefLE"), [
#            Bot(Race.Protoss, BinaryBot()),
#            Computer(Race.Terran, Difficulty.CheatVision)
#            ], realtime=False)

#    if diff == 8:
#        run_game(maps.get("AbyssalReefLE"), [
#            Bot(Race.Protoss, BinaryBot()),
#            Computer(Race.Terran, Difficulty.CheatMoney)
#            ], realtime=False)

#    if diff == 9:
#        run_game(maps.get("AbyssalReefLE"), [
#            Bot(Race.Protoss, BinaryBot()),
#            Computer(Race.Terran, Difficulty.CheatInsane)
#            ], realtime=False)

# if __name__ == '__main__':
#    main()