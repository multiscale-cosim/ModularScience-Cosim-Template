#  Copyright 2020 Forschungszentrum Jülich GmbH and Aix-Marseille Université
# "Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements; and to You under the Apache License, Version 2.0. "

import nest
import nest.raster_plot
import matplotlib.pyplot as plt
import os
import json
import sys

from cosim_example_demos.TVB_NEST_demo.tvb_sim.utils_tvb import create_logger
from cosim_example_demos.TVB_NEST_demo.nest_sim.utils_function import wait_transformation_modules, get_data
from EBRAINS_RichEndpoint.Application_Companion.common_enums import SteeringCommands
from EBRAINS_RichEndpoint.Application_Companion.common_enums import INTEGRATED_SIMULATOR_APPLICATION as SIMULATOR


class NESTAdapter:

    def __init__(self):
        path_to_self = os.path.dirname(__file__)
        path_to_parameters_file = path_to_self + '/../../result_sim/co-simulation/parameter.json'
        with open(path_to_parameters_file) as f:
            self.__parameters = json.load(f)
        if "time_synchronization" not in self.__parameters.keys():
            self.__parameters['time_synchronization'] = -1

        # self.__nest = nest
        # self.__running_command = SteeringCommands.INIT
        
        self.__initialize_parameters(
            co_simulation=self.__parameters['co_simulation'],
            path=self.__parameters['path'],
            simtime=self.__parameters['simulation_time'],
            resolution=self.__parameters['resolution'],
            time_synch=self.__parameters['time_synchronization'],
            nb_neurons=self.__parameters['nb_neurons'][0]
            )
        log_level = self.__parameters['level_log']
        self.__logger = create_logger(self.__parameters['path'], 'nest', log_level)
        self.__logger.info("initialized")

    def __initialize_parameters(self, co_simulation, path, time_synch=0.1, simtime=1000.0, resolution=0.1, nb_neurons=10000):
        # self.__running_command = SteeringCommands.INIT
        self.__cosimulaiton = co_simulation
        self.__path = path
        self.__simulation_time = simtime
        self.__resolution = resolution
        self.__time_synch = time_synch
        self.__nb_neurons = nb_neurons

    def __configure_nest(self, simulator, co_simulation, nb_neurons=10000):
        """
        configure NEST before the simulation
        modify example of https://simulator.simulator.readthedocs.io/en/stable/_downloads/482ad6e1da8dc084323e0a9fe6b2c7d1/brunel_alpha_simulator.py
        :param simulator: nest simulator
        :param co_simulation: boolean for checking if the co-simulation is active or not
        :param nb_neurons: number of neurons
        :return:
        """
        # create the neurons and the devices
        neuron_params = {"C_m": 250.0, "tau_m": 20.0, "tau_syn_ex": 0.5, "tau_syn_in": 0.5,
                        "t_ref": 2.0, "E_L": 0.0, "V_reset": 0.0, "V_m": 0.0, "V_th": 20.0}
        nodes_ex = simulator.Create("iaf_psc_alpha", nb_neurons, params=neuron_params)
        nodes_in = simulator.Create("iaf_psc_alpha", 25, params=neuron_params)
        noise = simulator.Create("poisson_generator", params={"rate": 8894.503857360944})
        espikes = simulator.Create("spike_recorder")
        ispikes = simulator.Create("spike_recorder")
        espikes.set(label="brunel-py-ex", record_to="ascii")
        ispikes.set(label="brunel-py-in", record_to="ascii")
        # create the connection
        simulator.CopyModel("static_synapse", "excitatory", {"weight": 20.68015524367846, "delay": 1.5})
        simulator.CopyModel("static_synapse", "inhibitory", {"weight": -103.4007762183923, "delay": 1.5})
        simulator.Connect(noise, nodes_ex, syn_spec="excitatory")
        simulator.Connect(noise, nodes_in, syn_spec="excitatory")
        simulator.Connect(nodes_ex[:50], espikes, syn_spec="excitatory")
        simulator.Connect(nodes_in[:25], ispikes, syn_spec="excitatory")
        conn_params_ex = {'rule': 'fixed_indegree', 'indegree': 10}
        conn_params_in = {'rule': 'fixed_indegree', 'indegree': 2}
        simulator.Connect(nodes_ex, nodes_ex + nodes_in, conn_params_ex, "excitatory")
        simulator.Connect(nodes_in, nodes_ex + nodes_in, conn_params_in, "inhibitory")
        # Cosimulation devices
        if co_simulation:
            input_to_simulator = simulator.Create("spike_generator", nb_neurons,
                                                params={'stimulus_source': 'mpi',
                                                        'label': '/../transformation/spike_generator'})
            output_from_simulator = simulator.Create("spike_recorder",
                                                    params={"record_to": "mpi",
                                                            'label': '/../transformation/spike_detector'})
            simulator.Connect(input_to_simulator, nodes_ex, {'rule': 'one_to_one'},
                            {"weight": 20.68015524367846, "delay": 0.1})
            simulator.Connect(nodes_ex, output_from_simulator, {'rule': 'all_to_all'},
                            {"weight": 1.0, "delay": 0.1})
            return espikes, input_to_simulator, output_from_simulator
        else:
            return espikes, None, None

    
    def execute_init_command(self):
        self.__logger.debug("executing INIT command")
        nest.ResetKernel()
        nest.SetKernelStatus(
            {"data_path": self.__path + '/nest/', "overwrite_files": True, "print_time": True, "resolution": self.__resolution})

        self.__logger.info("configure the network")
        espikes, input_to_simulator, output_from_simulator =\
             self.__configure_nest(nest, self.__cosimulaiton, self.__nb_neurons)

        self.__logger.info("establishing the connections")
        wait_transformation_modules(nest, self.__path, input_to_simulator, output_from_simulator, self.__logger)   
        self.__logger.info("preparing the simulator")
        nest.Prepare()
        self.__logger.info("connections are made") 
        self.__logger.debug("INIT command is executed")
        return self.__time_synch  # minimum step size for simulation


    def execute_start_command(self):
        self.__logger.debug("executing START command")
        count = 0.0
        self.__logger.debug('starting simulation')
        while count * self.__time_synch < self.__simulation_time:
            nest.Run(self.__time_synch)
            count += 1
            
        self.__logger.debug('nest simulation is finished')
        self.__logger.info("cleaning up NEST")
        nest.Cleanup()
        # self.execute_end_command()


    def execute_end_command(self):

        self.__logger.info("plotting the result")
        if nest.Rank() == 0:
            nest.raster_plot.from_data(get_data(self.__path + '/nest/'))
            plt.savefig(self.__path + "/figures/plot_nest.png")
        
        self.__logger.debug("post processing is done")

if __name__ == "__main__":
    
    nest_adapter = NESTAdapter()
    local_minimum_step_size = nest_adapter.execute_init_command()
     # send local minimum step size to Application Manager as a response to INIT
    # NOTE Application Manager expects a string in the following format:
    # {'PID': <int>, 'LOCAL_MINIMUM_STEP_SIZE': <float>}
    pid_and_local_minimum_step_size = \
        {SIMULATOR.PID.name: os.getpid(),
        SIMULATOR.LOCAL_MINIMUM_STEP_SIZE.name: local_minimum_step_size}

    # Application Manager will read the stdout stream via PIPE
    # NOTE the communication with Application Manager via PIPES will be
    # changed to some other mechanism
    print(f'{pid_and_local_minimum_step_size}')
    user_action_command = input()
    # execute if steering command is START
    if SteeringCommands[user_action_command] == SteeringCommands.START:
        nest_adapter.execute_start_command()
        nest_adapter.execute_end_command()
        sys.exit(0)
    else:
        # TODO raise and log the exception with traceback and terminate with
        # error if received an unknown steering command
        print(f'unknown steering command: '
              f'{SteeringCommands[user_action_command]}',
              file=sys.stderr)
        sys.exit(1)


    # def run_example(steering_command, co_simulation, path, time_synch=0.1, simtime=1000.0, level_log=1, resolution=0.1, nb_neurons=10000):
    #     """
    #     run the example for NEST
    #     :param steering_command: current steering command to be executed 
    #     :param co_simulation: boolean for checking if the co-simulation is active or not
    #     :param path: path of the result of the simulation
    #     # parameters for the co-simulation
    #     :param time_synch: time of synchronization between simulator
    #     :param simtime: time of simulation
    #     :param level_log: level of the log
    #     :param resolution: resolution of the simulation
    #     :param nb_neurons: number of neurons
    #     :return:
    #     """
    
    #     nest.ResetKernel()
    #     nest.SetKernelStatus(
    #         {"data_path": path + '/nest/', "overwrite_files": True, "print_time": True, "resolution": resolution})

    #     logger.info("configure the network")
    #     espikes, input_to_simulator, output_from_simulator = configure(nest, co_simulation, nb_neurons)

    #     logger.info("start the simulation")
    #     if not co_simulation:
    #         nest.Simulate(simtime)
    #     else:
    #         wait_transformation_modules(nest, path, input_to_simulator, output_from_simulator, logger)   
    #         count = 0.0
    #         nest.Prepare()
    #         logger.info("connections are made") 
    #         while count * time_synch < simtime:
    #             nest.Run(time_synch)
    #             logger.info("count is now: " + str(count))
    #             count += 1
    #         nest.Cleanup()

    #     logger.info("plot the result")
    #     if nest.Rank() == 0:
    #         nest.raster_plot.from_data(get_data(path + '/nest/'))
    #         plt.savefig(path + "/figures/plot_nest.png")




# if __name__ == "__main__":

#     if len(sys.argv) == 1:  # test the example
#         path_file = os.path.dirname(__file__)
#         create_folder(path_file + "/../../result_sim/nest_only/")
#         create_folder(path_file + "/../../result_sim/nest_only/log/")
#         create_folder(path_file + "/../../result_sim/nest_only/figures/")
#         create_folder(path_file + "/../../result_sim/nest_only/nest/")
#         run_example(False, path_file + "/../../result_sim/nest_only/")
#     elif len(sys.argv) == 2:  # run with parameters file
#         import json
#         from pathlib import Path
#         path_file = os.path.dirname(__file__)
#         path = path_file+ '/../../result_sim/co-simulation/parameter.json'
#         with open(path) as f:
#             parameters = json.load(f)
#         if "time_synchronization" not in parameters.keys():
#             parameters['time_synchronization'] = -1
#         run_example(parameters['co_simulation'], parameters['path'], simtime=parameters['simulation_time'],
#                     level_log=parameters['level_log'], resolution=parameters['resolution'],
#                     time_synch=parameters['time_synchronization'], nb_neurons=parameters['nb_neurons'][0])
#     elif len(sys.argv) == 6:  # run with parameter in command line
#         run_example(bool(int(sys.argv[1])), sys.argv[2], time_synch=float(sys.argv[3]),
#                     simtime=float(sys.argv[4]), level_log=int(sys.argv[5]))
