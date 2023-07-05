# ------------------------------------------------------------------------------
#  Copyright 2020 Forschungszentrum Jülich GmbH and Aix-Marseille Université
# "Licensed to the Apache Software Foundation (ASF) under one or more contributor
#  license agreements; and to You under the Apache License, Version 2.0. "
#
# Forschungszentrum Jülich
#  Institute: Institute for Advanced Simulation (IAS)
#    Section: Jülich Supercomputing Centre (JSC)
#   Division: High Performance Computing in Neuroscience
# Laboratory: Simulation Laboratory Neuroscience
#       Team: Multi-scale Simulation and Design
# ------------------------------------------------------------------------------

import os
import json
import copy
from common.utils import directory_utils
from EBRAINS_ConfigManager.global_configurations_manager.xml_parsers.default_directories_enum import DefaultDirectories


class SetupResultDirectories:
    '''
        sets up the cosimulationb parameters and directories to save the output
    '''
    def __init__(self, path):
        parameter_default = {"co_simulation": False,
                             "path": "",
                             "simulation_time": 1.0,
                             "level_log": 1,
                             "resolution": 0.1,
                             "nb_neurons": [100]
                            }

        ### NOTE: temporary result folder creation, change with refactoring
        path_file = path
        parameter_co_simulation = copy.copy(parameter_default)
        parameter_co_simulation['path'] = path_file
        parameter_co_simulation.update({
            "co_simulation": True,
            # parameter for the synchronization between simulators
            "time_synchronization": 1.2,
            "id_nest_region": [0],
            # parameter for the transformation of data between scale
            "nb_brain_synapses": 1,
            'id_first_neurons': [1],
            "save_spikes": True,
            "save_rate": True,
        })
        self.setup_directories(parameter_co_simulation)

    def setup_directories(self, parameters):
        '''
        run the simulation
        :param parameters: parameters of the simulation
        :return:
        '''
        path = parameters['path']
        # start to create the repertory for the simulation
        self.create_folder(path + '/nest')
        self.create_folder(path + '/tvb')
        self.create_folder(path + '/figures')
        self.save_parameter(parameters)

    def save_parameter(self, parameters):
        """
        save the parameters of the simulations in json file
        :param parameters: dictionary of parameters
        :return: nothing
        """
        # save the value of all parameters
        f = open(parameters['path'] + '/parameter.json', "wt")
        json.dump(parameters, f)
        f.close()

    def create_folder(self, path): return directory_utils.safe_makedir(path)
