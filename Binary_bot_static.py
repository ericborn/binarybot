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
#import keras
import math
import time
import csv
import os

# import asyncio
# from absl import app

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
# diff_list = [
#     'Difficulty.VeryEasy', 'Difficulty.Easy', 'Difficulty.Medium,'
#     'Difficulty.MediumHard', 'Difficulty.Hard', 'Difficulty.Harder',
#     'Difficulty.VeryHard', 'Difficulty.CheatMoney', 'Difficulty.CheatVision', 
#     'Difficulty.CheatInsane'
# ]

# maps the functions from the pysc2 actions file
FUNCTIONS = actions.FUNCTIONS

class BinaryBot(sc2.BotAI):
    def __init__(self):
        #self.ITERATIONS_PER_MINUTE = 500
        self.MAX_WORKERS = 50
        
        # path to save outcome and difficulty to csv
        self.csv_path = 'C:/Users/TomBrody/Desktop/School/767 ML/SC Bot/Q-learn/record.csv'
        self.text_path = 'C:/Users/TomBrody/Desktop/School/767 ML/SC Bot/Q-learn/record.txt'
        
        # Used to slow down the bots actions
        self.do_something_after = 0
        self.delay_time = 0
        self.delay = 25

        # used to allow a trained model to chose actions instead of random
        #$self.use_model = use_model
        
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

    # Create a function to write the result to a text file
    def write_txt(self, result, diff):
        with open(self.text_path,'a') as textfile:
            outcome = str(result) + ',' + str(diff) + '\n'
            textfile.write(outcome)
            textfile.close()

    def appender(self):
        # appends all supply data to the training_data list
        # then extends that list with the action, troop and outcome data
        # in the last index with -1
        # on the next loop it appends a new list to the origina list 
        # and repeats the process
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
        self.action_data = np.zeros(9)

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
                self.units(ROBOTICSFACILITY).amount, 
                self.units(STARGATE).amount, self.units(ROBOTICSBAY).amount,
                self.state.score.killed_value_structures,
                self.state.score.killed_value_units
            ])
            self.training_data[-1].extend(self.action_data)
            self.training_data[-1].extend(self.troop_data)
            self.training_data[-1].extend(self.outcome_data)

            self.write_txt(str(-1), 'Medium')
            #self.write_csv(str(-1))
            #self.write_txt(str(-1), diff_dict[diff])

#            np.save(r"C:/botdata/{}.npy".format(str(int(time.time()))),
#                    np.array(self.training_data))
        
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

            self.write_txt(str(-1), 'Medium')
            #self.write_csv(1)
            #self.write_txt(str(1), diff_dict[diff])
            
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

            self.write_txt(str(-1), 'Medium')         
            #self.write_csv(0)
            #self.write_txt(str(0), diff_dict[diff])
            
#            np.save(r"C:/botdata/{}.npy".format(str(int(time.time()))),
#                    np.array(self.training_data))

    # This is the function steps forward and is called through each frame of the game
    async def on_step(self, iteration):
        self.iteration = iteration

        await self.attack()
        await self.expand()
        await self.do_nothing()
        await self.back_to_work()
        await self.build_pylons()
        await self.build_workers()
        await self.build_assimilators()
        await self.distribute_workers()
        await self.build_offensive_force()
        await self.offensive_force_buildings()

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

    # attempt to fix workers starting the warp in of a building
    # and not going back to work until its finished.
    # checks for idle workers then calls a distribute_workers
    # to send them back to work.
    # Does not work on workers who create assimilators since they're
    # being assigned to get gas upon starting the build
    async def back_to_work(self):
        if self.idle_worker_count > 0:
            self.distribute_workers

    def find_target(self, state):
        if len(self.known_enemy_units) > 0:
            return random.choice(self.known_enemy_units)
        elif len(self.known_enemy_structures) > 0:
            return random.choice(self.known_enemy_structures)
        else:
            return self.enemy_start_locations[0]

    # Action 1 - Attack
    async def attack(self):
        # print('attack')
        attack_amount = random.randrange(6, 10)
        # if self.units.of_type([ZEALOT, STALKER, ADEPT, IMMORTAL, VOIDRAY, 
        #                        COLOSSUS]).amount
        if len(self.units.of_type([ZEALOT, STALKER, ADEPT, IMMORTAL, VOIDRAY, 
                               COLOSSUS]).idle) > attack_amount:
            self.action_data[0] = 1
            self.appender()
            for s in self.units.of_type([ZEALOT, STALKER, ADEPT, IMMORTAL, 
                                         VOIDRAY, COLOSSUS]).idle:
                await self.do(s.attack(self.find_target(self.state)))

    # Action 2 - build assimilators
    # TODO
    # need to add check to move probes onto gas at this same step
    async def build_assimilators(self):
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
                        self.action_data[1] = 1
                        self.appender()
                        await self.do(worker.build(ASSIMILATOR, vaspene))


    # Action 3 - build offensive force
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
        
        #self.action_data = np.zeros(8)
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
        # if unit_choice == 'ZEALOT' and self.can_afford(ZEALOT) and \
        # self.supply_left >= 2:
        #     self.troop_data[0] = 1
        #     self.action_data[2] = 1
        #     self.appender()
        #     for gw in self.units(GATEWAY).ready.idle:
        #         await self.do(gw.train(ZEALOT)) 

        if unit_choice == 'ZEALOT' and self.can_afford(STALKER) and \
        self.supply_left >= 2:
            self.troop_data[0] = 1
            self.action_data[2] = 1
            self.appender()
            for gw in self.units(GATEWAY).ready.idle:
                await self.do(gw.train(STALKER))  

        elif unit_choice == 'STALKER' and self.can_afford(STALKER) and \
        self.supply_left >= 2:
            self.troop_data[1] = 1
            self.action_data[2] = 1
            self.appender()
            for gw in self.units(GATEWAY).ready.idle:
                await self.do(gw.train(STALKER))
                await self.do(gw.train(STALKER))

        elif unit_choice == 'ADEPT' and self.can_afford(ADEPT) and \
        self.supply_left >= 2:
            self.troop_data[2] = 1
            self.action_data[2] = 1
            self.appender()
            for gw in self.units(GATEWAY).ready.idle:
                await self.do(gw.train(ADEPT))

        elif unit_choice == 'IMMORTAL' and self.can_afford(IMMORTAL) and \
        self.supply_left >= 4:
            self.troop_data[3] = 1
            self.action_data[2] = 1
            self.appender()
            for gw in self.units(ROBOTICSFACILITY).ready.idle:
                await self.do(gw.train(IMMORTAL))

        elif unit_choice == 'VOIDRAY' and self.can_afford(VOIDRAY) and \
        self.supply_left >= 4:
            self.troop_data[4] = 1
            self.action_data[2] = 1
            self.appender()
            for gw in self.units(STARGATE).ready.idle:
                await self.do(gw.train(VOIDRAY))

        elif unit_choice == 'COLOSSUS' and self.can_afford(VOIDRAY) and \
        self.supply_left >= 6:
            self.action_data[2] = 1
            self.troop_data[5] = 1
            self.appender()
            for gw in self.units(STARGATE).ready.idle:
                await self.do(gw.train(VOIDRAY))
                await self.do(gw.train(VOIDRAY))

        # elif unit_choice == 'COLOSSUS' and self.can_afford(COLOSSUS) and \
        # self.supply_left >= 6:
        #     self.action_data[2] = 1
        #     self.troop_data[5] = 1
        #     self.appender()
        #     for gw in self.units(ROBOTICSFACILITY).ready.idle:
        #         await self.do(gw.train(COLOSSUS))
        #         await self.do(gw.train(COLOSSUS))
        #         await self.do(gw.train(COLOSSUS))
        # commented out, was building too many pylons
        #else:
        #    await self.build_pylons()

    async def do_nothing(self):
        #print('do_nothing')
        self.action_data[6] = 1
        wait = random.randrange(10, 30)/100
        self.appender()
        #self.do_something_after = self.time_loop + wait

    # builds 16 workers per nexus up to a maximum of 50
    async def build_workers(self):
        #print('build_workers')
        if (self.units(NEXUS).amount * 16) > self.units(PROBE).amount and \
            self.units(PROBE).amount < self.MAX_WORKERS:
            self.action_data[4] = 1
            self.appender()
            for nexus in self.units(NEXUS).ready.idle:
                if self.can_afford(PROBE):
                    await self.do(nexus.train(PROBE))

    async def build_pylons(self):
        #print('build_pylons')
        if self.supply_cap != 200 and self.supply_left < 10: 
            #and not self.already_pending(PYLON): #dont care, build another
            nexuses = self.units(NEXUS).ready
            if nexuses.exists:
                if self.can_afford(PYLON):
                    self.action_data[3] = 1
                    self.appender()
                    await self.build(PYLON, near=
                                     self.units(NEXUS).first.position.towards(
                                             self.game_info.map_center, 5))
    
    # Added not already_pending trying to prevent multiple
    # being built right next to each other
    async def expand(self):
        #print('expand')
        if self.can_afford(NEXUS) and not self.already_pending(NEXUS):
            self.action_data[7] = 1
            self.appender()
            await self.expand_now()

    async def offensive_force_buildings(self):
        #print('offensive_force_buildings')
        # Checks for a pylon as an indicator of where to build
        # small area around pylon is needed to place another building
        if self.units(PYLON).ready.exists:
            pylon = self.units(PYLON).ready.random

            # Gateway required first
            if self.can_afford(GATEWAY) and \
                not self.already_pending(GATEWAY):
                self.action_data[8] = 1
                self.appender()
                #and self.units(GATEWAY).amount <= 2:
                await self.build(GATEWAY, near=pylon)

            if self.units(GATEWAY).ready.exists and \
               self.units(CYBERNETICSCORE).amount < 1: # Added to limit to 1
                if self.can_afford(CYBERNETICSCORE) and not \
                   self.already_pending(CYBERNETICSCORE):
                    self.action_data[8] = 1
                    self.appender()
                    await self.build(CYBERNETICSCORE, near=pylon)

            if self.units(CYBERNETICSCORE).ready.exists:
                if self.can_afford(ROBOTICSFACILITY) and not \
                    self.already_pending(ROBOTICSFACILITY):
                    self.action_data[8] = 1
                    self.appender()
                    await self.build(ROBOTICSFACILITY, near=pylon)
            
            if self.units(CYBERNETICSCORE).ready.exists:
                if self.can_afford(STARGATE) and not \
                    self.already_pending(STARGATE):
                    self.action_data[8] = 1
                    self.appender()
                    await self.build(STARGATE, near=pylon)

            if self.units(ROBOTICSFACILITY).ready.exists:
                if self.can_afford(ROBOTICSBAY) and not \
                    self.already_pending(ROBOTICSBAY):
                    self.action_data[8] = 1
                    self.appender()
                    await self.build(ROBOTICSBAY, near=pylon)

# Fixed difficulty
def main():
    run_game(maps.get("AbyssalReefLE"), [
        Bot(Race.Protoss, BinaryBot()),
        Computer(Race.Terran, Difficulty.Medium)
        ], realtime=False)

if __name__ == '__main__':
    main()

# Random difficulty
# def main():
#     # Creates a random number between 0-9
#     diff = random.randrange(0,10)

#     # depending on the number selected a difficulty is chosen
#     if diff == 0:
#         run_game(maps.get("AbyssalReefLE"), [
#             Bot(Race.Protoss, BinaryBot()),
#             Computer(Race.Terran, Difficulty.VeryEasy)
#             ], realtime=False)

#     if diff == 1:
#         run_game(maps.get("AbyssalReefLE"), [
#             Bot(Race.Protoss, BinaryBot()),
#             Computer(Race.Terran, Difficulty.Easy)
#             ], realtime=False)
    
#     if diff == 2:
#         run_game(maps.get("AbyssalReefLE"), [
#             Bot(Race.Protoss, BinaryBot()),
#             Computer(Race.Terran, Difficulty.Medium)
#             ], realtime=False)

#     if diff == 3:
#         run_game(maps.get("AbyssalReefLE"), [
#             Bot(Race.Protoss, BinaryBot()),
#             Computer(Race.Terran, Difficulty.MediumHard)
#             ], realtime=False)
    
#     if diff == 4:
#         run_game(maps.get("AbyssalReefLE"), [
#             Bot(Race.Protoss, BinaryBot()),
#             Computer(Race.Terran, Difficulty.Hard)
#             ], realtime=False)

#     if diff == 5:
#         run_game(maps.get("AbyssalReefLE"), [
#             Bot(Race.Protoss, BinaryBot()),
#             Computer(Race.Terran, Difficulty.Harder)
#             ], realtime=False)
    
#     if diff == 6:
#         run_game(maps.get("AbyssalReefLE"), [
#             Bot(Race.Protoss, BinaryBot()),
#             Computer(Race.Terran, Difficulty.VeryHard)
#             ], realtime=False)

#     if diff == 7:
#         run_game(maps.get("AbyssalReefLE"), [
#             Bot(Race.Protoss, BinaryBot()),
#             Computer(Race.Terran, Difficulty.CheatVision)
#             ], realtime=False)

#     if diff == 8:
#         run_game(maps.get("AbyssalReefLE"), [
#             Bot(Race.Protoss, BinaryBot()),
#             Computer(Race.Terran, Difficulty.CheatMoney)
#             ], realtime=False)

#     if diff == 9:
#         run_game(maps.get("AbyssalReefLE"), [
#             Bot(Race.Protoss, BinaryBot()),
#             Computer(Race.Terran, Difficulty.CheatInsane)
#             ], realtime=False)

# if __name__ == '__main__':
#     main()