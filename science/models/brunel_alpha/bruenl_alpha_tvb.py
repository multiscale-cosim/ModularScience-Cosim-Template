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

from EBRAINS_ConfigManager.global_configurations_manager.xml_parsers.default_directories_enum import DefaultDirectories

import tvb.simulator.lab as lab
import matplotlib.pyplot as plt
from tvb.contrib.cosimulation.cosimulator import CoSimulator
from tvb.contrib.cosimulation.cosim_monitors import CosimCoupling


class BrunelAlphaTVB:
    def __init__(self, p_configurations_manager, p_log_settings,
                 sci_params):
            self._log_settings = p_log_settings
            self._configurations_manager = p_configurations_manager
            self.__logger = self._configurations_manager.load_log_configurations(
                name="BrunelAlphaTVB",
                log_configurations=self._log_settings,
                target_directory=DefaultDirectories.SIMULATION_RESULTS)
            self.__sci_params = sci_params

    def configure(self):
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