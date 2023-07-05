# ------------------------------------------------------------------------------
#  Copyright 2020 Forschungszentrum Jülich GmbH and Aix-Marseille Université
# "Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements; and to You under the Apache License,
# Version 2.0. "
#
# Forschungszentrum Jülich
# Institute: Institute for Advanced Simulation (IAS)
# Section: Jülich Supercomputing Centre (JSC)
# Division: High Performance Computing in Neuroscience
# Laboratory: Simulation Laboratory Neuroscience
# Team: Multi-scale Simulation and Design
# ------------------------------------------------------------------------------
import json
import os
import time


class Parameters:
    def __init__(self, path):
        self.__path_to_parameters_file = self.__get_path_to_parameters_file(path)
        self.__cosim_parameters = self.__load_parameters_from_json()
        self.__cosimulation = self.__cosim_parameters['co_simulation']
        self.__path = self.__cosim_parameters['path']
        self.__time_synch = self.__cosim_parameters['time_synchronization']
        self.__simulation_time = self.__cosim_parameters['simulation_time']
        self.__resolution = self.__cosim_parameters['resolution']
        self.__id_nest_region = self.__cosim_parameters['id_nest_region']
        self.__nb_neurons = self.__cosim_parameters['nb_neurons'][0]
        self.__log_level = self.__cosim_parameters['level_log']
        self.__set_nest_model_parameters()

    @property
    def cosim_parameters(self): return self.__cosim_parameters

    @property
    def co_simulaiton(self): return self.__cosimulation

    @property
    def path(self): return self.__path

    @property
    def time_synch(self): return self.__time_synch

    @property
    def simulation_time(self): return self.__simulation_time

    @property
    def resolution(self): return self.__resolution

    @property
    def id_nest_region(self): return self.__id_nest_region

    @property
    def nb_neurons(self): return self.__nb_neurons

    @property
    def neuron_params(self): return self.__neuron_params

    @property
    def nodes_model(self): return self.__nodes_model

    @property
    def noise_model(self): return self.__noise_model

    @property
    def noise_params(self): return self.__noise_params

    @property
    def excitatory_spikes_model(self): return self.__excitatory_spikes_model

    @property
    def inhibitory_spikes_model(self): return self.__inhibitory_spikes_model

    @property
    def spike_recorder_device(self): return self.__spike_recorder_device

    @property
    def predefined_synapse(self): return self.__predefined_synapse

    @property
    def customary_excitatory_synapse(self): return self.__customary_excitatory_synapse

    @property
    def customary_inhibitory_synapse(self): return self.__customary_inhibitory_synapse

    @property
    def excitatory_connection_params(self): return self.__excitatory_connection_params

    @property
    def inhibitory_connection_params(self): return self.__inhibitory_connection_params

    @property
    def connection_param_ex(self): return self.__connection_param_ex

    @property
    def connection_param_in(self): return self.__connection_param_in

    @property
    def total_inhibitory_nodes(self): return self.__total_inhibitory_nodes

    @property
    def log_level(self): return self.__log_level

    def __get_path_to_parameters_file(self, path):
        # path_to_self = os.path.dirname(__file__)
        return (path + '/parameter.json')

    def __load_parameters_from_json(self):
        # check if file is already created
        while not os.path.exists(self.__path_to_parameters_file):
            print(f'{self.__path_to_parameters_file} does not exist yet, retrying in 1 second')
            time.sleep(1)

        # file is already created
        with open(self.__path_to_parameters_file) as f:
            return json.load(f)

    def __set_nest_model_parameters(self):
        # neurons and the devices
        # self.__neuron_params = {"C_m": 250.0, "tau_m": 20.0, "tau_syn_ex": 0.5, "tau_syn_in": 0.5,
        #                 "t_ref": 2.0, "E_L": 0.0, "V_reset": 0.0, "V_m": 0.0, "V_th": 20.0}
        # self.__nodes_model = "iaf_psc_alpha"
        # self.__total_inhibitory_nodes = 25
        # self.__noise_model = "poisson_generator"
        # self.__noise_params = {"rate": 8894.503857360944}
        # self.__excitatory_spikes_model="brunel-py-ex"
        # self.__inhibitory_spikes_model="brunel-py-in"
        # self.__spike_recorder_device = "spike_recorder"
        # connection
        # self.__predefined_synapse = "static_synapse"
        # self.__customary_excitatory_synapse = "excitatory"
        # self.__customary_inhibitory_synapse = "inhibitory"
        # self.__excitatory_connection_params = {"weight": 20.68015524367846, "delay": 1.5}
        # self.__inhibitory_connection_params = {"weight": -103.4007762183923, "delay": 1.5}
        # self.__connection_param_ex = {'rule': 'fixed_indegree', 'indegree': 10}
        # self.__connection_param_in = {'rule': 'fixed_indegree', 'indegree': 2}
        pass
