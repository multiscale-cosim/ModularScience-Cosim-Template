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
import numpy
import sys
import os
import pickle
import base64
import ast

from action_adapters_alphabrunel.tvb_simulator.wrapper_TVB_mpi import TVBMpiWrapper
from action_adapters_alphabrunel.parameters import Parameters
from action_adapters_alphabrunel.resource_usage_monitor_adapter import ResourceMonitorAdapter
from common.utils.security_utils import check_integrity

from EBRAINS_RichEndpoint.application_companion.common_enums import SteeringCommands, COMMANDS
from EBRAINS_RichEndpoint.application_companion.common_enums import INTEGRATED_SIMULATOR_APPLICATION as SIMULATOR
from EBRAINS_RichEndpoint.application_companion.common_enums import INTEGRATED_INTERSCALEHUB_APPLICATION as INTERSCALE_HUB
from EBRAINS_ConfigManager.global_configurations_manager.xml_parsers.default_directories_enum import DefaultDirectories
from EBRAINS_ConfigManager.global_configurations_manager.xml_parsers.configurations_manager import ConfigurationsManager
from EBRAINS_ConfigManager.workflow_configurations_manager.xml_parsers.xml2class_parser import Xml2ClassParser
from EBRAINS_InterscaleHUB.Interscale_hub.interscalehub_enums import DATA_EXCHANGE_DIRECTION

import tvb.simulator.lab as lab
import matplotlib.pyplot as plt
from tvb.contrib.cosimulation.cosimulator import CoSimulator
from tvb.contrib.cosimulation.cosim_monitors import CosimCoupling

numpy.random.seed(125)


class TVBAdapter:

    def __init__(self, p_configurations_manager, p_log_settings,
                 p_interscalehub_addresses,
                 is_monitoring_enabled,
                 p_sci_params_xml_path_filename=None):
        self._log_settings = p_log_settings
        self._configurations_manager = p_configurations_manager
        self.__logger = self._configurations_manager.load_log_configurations(
            name="TVB_Adapter",
            log_configurations=self._log_settings,
            target_directory=DefaultDirectories.SIMULATION_RESULTS)
        self.__path_to_parameters_file = self._configurations_manager.get_directory(
            directory=DefaultDirectories.SIMULATION_RESULTS)

        # Load scientific parameters into an object
        self.__sci_params = Xml2ClassParser(p_sci_params_xml_path_filename, self.__logger)
        
        self.__parameters = Parameters(self.__path_to_parameters_file)
        self.__simulator_tvb = None
        self.__tvb_mpi_wrapper = None
        self.__my_pid = os.getpid()
        self.__is_monitoring_enabled = is_monitoring_enabled
        if self.__is_monitoring_enabled:
            self.__resource_usage_monitor = ResourceMonitorAdapter(self._configurations_manager,
                                                               self._log_settings,
                                                               self.pid,
                                                               "TVB")
        # initialize port_names of the Interscalehubs
        self.__init_port_names(p_interscalehub_addresses)
        self.__logger.debug(f"host_name:{os.uname()}")
        self.__logger.info("initialized")

    @property
    def pid(self):
        return self.__my_pid

    def __init_port_names(self, interscalehub_addresses):
        '''
        helper function to initialize the port_names
        '''
        self.__logger.debug("Interscalehubs endpoints: "
                            f" {interscalehub_addresses}")

        for interscalehub in interscalehub_addresses:
            self.__logger.debug(f"running interscalehub: {interscalehub}")
            # NEST_TO_TVB RECEIVER endpoint
            if interscalehub.get(INTERSCALE_HUB.DATA_EXCHANGE_DIRECTION.name) == DATA_EXCHANGE_DIRECTION.NEST_TO_TVB.name:
                self.__interscalehub_nest_to_tvb_address =\
                    interscalehub.get(INTERSCALE_HUB.MPI_CONNECTION_INFO.name)
                self.__logger.debug(f"Interscalehub_nest_to_tvb_address: {self.__interscalehub_nest_to_tvb_address}")

            # TVB_TO_NEST SENDER endpoint
            elif interscalehub.get(INTERSCALE_HUB.DATA_EXCHANGE_DIRECTION.name) == DATA_EXCHANGE_DIRECTION.TVB_TO_NEST.name:
                self.__interscalehub_tvb_to_nest_address =\
                    interscalehub.get(INTERSCALE_HUB.MPI_CONNECTION_INFO.name)
                self.__logger.debug(f"Interscalehub_tvb_to_nest_address: {self.__interscalehub_tvb_to_nest_address}")

    # def __configure(self, time_synch=0.1, id_nest_region=None, dt=0.1):
    def __configure(self):
        """
        configure TVB before the simulation
        modify example of https://github.com/the-virtual-brain/tvb-root/blob/master/tvb_documentation/tutorials/tutorial_s1_region_simulation.ipynb
        based on: https://github.com/multiscale-cosim/TVB-NEST-demo/blob/main/nest_elephant_tvb/launcher/run.py
        :param: no parameters required
        :return: simulator
        """
        # :param time_synch: time of synchronization between simulator
        # :param id_nest_region: id of the region simulated with NEST
        # :param dt: size of the integration step
        # :return:
        # """
        oscillator = lab.models.Generic2dOscillator()
        #
        white_matter = lab.connectivity.Connectivity.from_file()
        # white_matter.speed = numpy.array([4.0])
        white_matter.speed = self.__sci_params.white_matter_speed
        #
        # white_matter_coupling = lab.coupling.Linear(a=numpy.array([0.154]))
        white_matter_coupling = lab.coupling.Linear(a=self.__sci_params.lab_coupling_linear_a)
        #
        # heunint = lab.integrators.HeunDeterministic(dt=dt)
        heunint = lab.integrators.HeunDeterministic(dt=self.__sci_params.heun_deterministic_dt)
        #
        what_to_watch = (lab.monitors.Raw(),)
        # special monitor for MPI
        simulator = CoSimulator(
            voi=numpy.array([0]),  # coupling with Excitatory firing rate
            synchronization_time=self.__sci_params.synchronization_time,  #  synchronization_time=time_synch,  # time of synchronization time between simulators
            # monitor for the coupling between simulators
            cosim_monitors=(CosimCoupling(coupling=white_matter_coupling),),
            proxy_inds=self.__sci_params.proxy_inds,   #  proxy_inds=numpy.array(id_nest_region, dtype=int),  # id of the proxy node
            model=oscillator,
            connectivity=white_matter,
            coupling=white_matter_coupling,
            integrator=heunint,
            monitors=what_to_watch
        )
        simulator.configure()
        self.__logger.info(f'TVB simulator has been configured...')
        return simulator

    def execute_init_command(self):
        self.__logger.debug("executing INIT command")
        self.__simulator_tvb = self.__configure()
        self.__simulator_tvb.simulation_length = self.__parameters.simulation_time
        # set up MPI connections
        self.__tvb_mpi_wrapper = TVBMpiWrapper(
            self._log_settings,
            self._configurations_manager,
            self.__simulator_tvb,
            intercalehub_nest_to_tvb=self.__interscalehub_nest_to_tvb_address,
            intercalehub_tvb_to_nest=self.__interscalehub_tvb_to_nest_address)
        self.__tvb_mpi_wrapper.init_mpi()
        self.__logger.debug("INIT command is executed")
        return self.__parameters.time_synch  # minimum step size for simulation 

    def execute_start_command(self, global_minimum_step_size):
        self.__logger.debug("executing START command")
        if self.__is_monitoring_enabled:
            self.__resource_usage_monitor.start_monitoring()
        self.__logger.debug(f'global_minimum_step_size: {global_minimum_step_size}')
        (r_raw_results,) = self.__tvb_mpi_wrapper.run_simulation_and_data_exchange(global_minimum_step_size)
        self.__logger.debug('TVB simulation is finished')
        return r_raw_results

    def execute_end_command(self, p_raw_results=None):
        if self.__is_monitoring_enabled:
            self.__resource_usage_monitor.stop_monitoring()
        self.__logger.info("plotting the result")
        plt.figure(1)
        plt.plot(p_raw_results[0], raw_results[1][:, 0, :, 0] + 3.0)
        plt.title("Raw -- State variable 0")
        plt.savefig(self.__parameters.path + "/figures/plot_tvb.png")
        self.__logger.debug("post processing is done")


if __name__ == "__main__":
    # TODO better handling of arguments parsing
    if len(sys.argv) == 6:
        # 1. parse arguments
        # unpickle configurations_manager object
        configurations_manager = pickle.loads(base64.b64decode(sys.argv[1]))
        # unpickle log_settings
        log_settings = pickle.loads(base64.b64decode(sys.argv[2]))
        # get science parameters XML file path
        p_sci_params_xml_path_filename = sys.argv[3]
        # flag indicating whether resource usage monitoring is enabled
        is_monitoring_enabled = pickle.loads(base64.b64decode(sys.argv[4]))
        # get interscalehub connection details
        p_interscalehub_address = pickle.loads(base64.b64decode(sys.argv[5]))

        # 2. security check of pickled objects
        # it raises an exception, if the integrity is compromised
        check_integrity(configurations_manager, ConfigurationsManager)
        check_integrity(log_settings, dict)
        check_integrity(p_interscalehub_address, list)
        check_integrity(is_monitoring_enabled, bool)

        # 3. everything is fine, configure simulator
        tvb_adapter = TVBAdapter(
            configurations_manager,
            log_settings,
            p_interscalehub_address,
            is_monitoring_enabled,
            p_sci_params_xml_path_filename=p_sci_params_xml_path_filename)
        
        # 4. execute 'INIT' command which is implicit with when laucnhed
        local_minimum_step_size = tvb_adapter.execute_init_command()
        
        # 5. send the pid and the local minimum step size to Application Manager
        # as a response to 'INIT' as per protocol
        
        # NOTE Application Manager expects a string in the following format:
        # {'PID': <pid>, 'LOCAL_MINIMUM_STEP_SIZE': <step size>}

        # prepare the response
        pid_and_local_minimum_step_size = \
            {SIMULATOR.PID.name: tvb_adapter.pid,
            # SIMULATOR.PID.name: os.getpid(),
            SIMULATOR.LOCAL_MINIMUM_STEP_SIZE.name: local_minimum_step_size}
        
        # send the response
        # NOTE Application Manager will read the stdout stream via PIPE
        print(f'{pid_and_local_minimum_step_size}')

        # 6. fetch next command from Application Manager
        user_action_command = input()

        # NOTE Application Manager sends the control commands with parameters in
        # the following specific format as a string via stdio:
        # {'STEERING_COMMAND': {'<Enum SteeringCommands>': <Enum value>}, 'PARAMETERS': <value>}
        
        # For example:
        # {'STEERING_COMMAND': {'SteeringCommands.START': 2}, 'PARAMETERS': 1.2}        

        # convert the received string to dictionary
        control_command = ast.literal_eval(user_action_command.strip())
        # get steering command
        steering_command_dictionary = control_command.get(COMMANDS.STEERING_COMMAND.name)
        current_steering_command = next(iter(steering_command_dictionary.values()))
        
        # 7. execute if steering command is 'START'
        if current_steering_command == SteeringCommands.START:
            # fetch global minimum step size
            global_minimum_step_size = control_command.get(COMMANDS.PARAMETERS.name)
            # execute the command
            raw_results = tvb_adapter.execute_start_command(global_minimum_step_size[0])
            tvb_adapter.execute_end_command(raw_results)
            # exit with success code
            sys.exit(0)
        else:
            print(f'unknown command: {current_steering_command}', file=sys.stderr)
            sys.exit(1)
    else:
        print(f'missing argument[s]; required: 5, received: {len(sys.argv)}',
              file=sys.stderr)
        print(f'Argument list received: {str(sys.argv)}', file=sys.stderr)
        sys.exit(1)
