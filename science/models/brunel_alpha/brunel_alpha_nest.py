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
"""
    Random balanced network HPC benchmark NEST
    Based on example: https://nest-simulator.readthedocs.io/en/v3.4/auto_examples/hpc_benchmark.html#sphx-glr-auto-examples-hpc-benchmark-py

    provides with the functions to build and simulate the Neural Network with NEST
"""
import numpy as np
import os
import sys
import time
import scipy.special as sp

import nest
import nest.raster_plot

from science.parameters.brunel_alpha.model_parameters import Parameters

from EBRAINS_ConfigManager.global_configurations_manager.xml_parsers.default_directories_enum import DefaultDirectories

M_INFO = 10
M_ERROR = 30


class BrunelAlphaHPC:
    def __init__(self,
                 log_settings,
                 configurations_manager,
                 interscalehub_tvb_to_nest_address,
                 interscalehub_nest_to_tvb_address):

        self.__logger = configurations_manager.load_log_configurations(
            name="BruenlAlphaModel",
            log_configurations=log_settings,
            target_directory=DefaultDirectories.SIMULATION_RESULTS)

        self.__params = Parameters()
        self.__brunel_params = self.__params.brunel_params
        self.__sim_params = self.__params.sim_parameters
        self.__interscalehub_nest_to_tvb_address = interscalehub_nest_to_tvb_address
        self.__interscalehub_tvb_to_nest_address = interscalehub_tvb_to_nest_address

        self.__logger.debug("initialized")

    def build_network(self):
        """Builds the network including setting of simulation and neuron
        parameters, creation of neurons and connections

        Requires an instance of Logger as argument

        """

        tic = time.time()  # start timer on construction

        # 1. unpack a few variables for convenience
        NE = self.__brunel_params['NE']  # number of excitatory neurons
        NI = self.__brunel_params['NI']  # number of inhibitory neurons
        model_params = self.__brunel_params['model_params']
        stdp_params = self.__brunel_params['stdp_params']

        # 2. set global kernel parameters
        nest.total_num_virtual_procs = self.__sim_params['nvp']
        nest.resolution = self.__sim_params['dt']
        nest.overwrite_files = True

        # 3. create populations
        nest.message(M_INFO, 'build_network', 'Creating excitatory population.')
        E_neurons = nest.Create('iaf_psc_alpha', NE, params=model_params)

        nest.message(M_INFO, 'build_network', 'Creating inhibitory population.')
        I_neurons = nest.Create('iaf_psc_alpha', NI, params=model_params)

        if self.__brunel_params['randomize_Vm']:
            nest.message(M_INFO, 'build_network',
                        'Randomizing membrane potentials.')

            random_vm = nest.random.normal(self.__brunel_params['mean_potential'],
                                        self.__brunel_params['sigma_potential'])
            nest.GetLocalNodeCollection(E_neurons).V_m = random_vm
            nest.GetLocalNodeCollection(I_neurons).V_m = random_vm

        # number of incoming excitatory connections
        CE = int(1. * NE / self.__sim_params['scale'])
        # number of incomining inhibitory connections
        CI = int(1. * NI / self.__sim_params['scale'])

        nest.message(M_INFO, 'build_network',
                    'Creating excitatory stimulus generator.')

        # 4. Convert synapse weight from mV to pA
        conversion_factor = self.__params.convert_synapse_weight(
            model_params['tau_m'], model_params['tau_syn_ex'], model_params['C_m'])
        JE_pA = conversion_factor * self.__brunel_params['JE']

        nu_thresh = model_params['V_th'] / (
            CE * model_params['tau_m'] / model_params['C_m'] *
            JE_pA * np.exp(1.) * self.__params.tau_syn)
        nu_ext = nu_thresh * self.__brunel_params['eta']

        # E_stimulus = nest.Create('poisson_generator', 1, {
        #                         'rate': nu_ext * CE * 1000.})
        E_stimulus = nest.Create("spike_generator", NE,
                                 params={'stimulus_source': 'mpi',
                                 'label': self.__interscalehub_tvb_to_nest_address})
        # create spike recorders
        nest.message(M_INFO, 'build_network',
                    'Creating excitatory spike recorder.')

        if self.__sim_params['record_spikes']:
            # recorder_label = os.path.join(
            #     self.__brunel_params['filestem'],
            #     'alpha_' + str(stdp_params['alpha']) + '_spikes')
            E_recorder = nest.Create('spike_recorder', params={
                'record_to': 'mpi',
                'label': self.__interscalehub_nest_to_tvb_address
            })

        BuildNodeTime = time.time() - tic

        self.__logger.info(str(BuildNodeTime) + ' # build_time_nodes')
        self.__logger.info(str(memory_thisjob()) + ' # virt_mem_after_nodes')

        tic = time.time()

        nest.SetDefaults('static_synapse_hpc', {'delay': self.__brunel_params['delay']})
        nest.CopyModel('static_synapse_hpc', 'syn_ex',
                    {'weight': JE_pA})
        nest.CopyModel('static_synapse_hpc', 'syn_in',
                    {'weight': self.__brunel_params['g'] * JE_pA})

        stdp_params['weight'] = JE_pA
        nest.SetDefaults('stdp_pl_synapse_hom_hpc', stdp_params)

        nest.message(M_INFO, 'build_network', 'Connecting stimulus generators.')

        # Connect Poisson generator to neuron

        nest.Connect(E_stimulus, E_neurons, {'rule': 'all_to_all'},
                    {'synapse_model': 'syn_ex'})
        nest.Connect(E_stimulus, I_neurons, {'rule': 'all_to_all'},
                    {'synapse_model': 'syn_ex'})

        nest.message(M_INFO, 'build_network',
                    'Connecting excitatory -> excitatory population.')

        nest.Connect(E_neurons, E_neurons,
                    {'rule': 'fixed_indegree', 'indegree': CE,
                    'allow_autapses': False, 'allow_multapses': True},
                    {'synapse_model': 'stdp_pl_synapse_hom_hpc'})

        nest.message(M_INFO, 'build_network',
                    'Connecting inhibitory -> excitatory population.')

        nest.Connect(I_neurons, E_neurons,
                    {'rule': 'fixed_indegree', 'indegree': CI,
                    'allow_autapses': False, 'allow_multapses': True},
                    {'synapse_model': 'syn_in'})

        nest.message(M_INFO, 'build_network',
                    'Connecting excitatory -> inhibitory population.')

        nest.Connect(E_neurons, I_neurons,
                    {'rule': 'fixed_indegree', 'indegree': CE,
                    'allow_autapses': False, 'allow_multapses': True},
                    {'synapse_model': 'syn_ex'})

        nest.message(M_INFO, 'build_network',
                    'Connecting inhibitory -> inhibitory population.')

        nest.Connect(I_neurons, I_neurons,
                    {'rule': 'fixed_indegree', 'indegree': CI,
                    'allow_autapses': False, 'allow_multapses': True},
                    {'synapse_model': 'syn_in'})

        if self.__sim_params['record_spikes']:
            if self.__sim_params['nvp'] != 1:
                local_neurons = nest.GetLocalNodeCollection(E_neurons)
                # GetLocalNodeCollection returns a stepped composite NodeCollection, which
                # cannot be sliced. In order to allow slicing it later on, we're creating a
                # new regular NodeCollection from the plain node IDs.
                local_neurons = nest.NodeCollection(local_neurons.tolist())
            else:
                local_neurons = E_neurons

            if len(local_neurons) < self.__brunel_params['Nrec']:
                nest.message(
                    M_ERROR, 'build_network',
                    """Spikes can only be recorded from local neurons, but the
                    number of local neurons is smaller than the number of neurons
                    spikes should be recorded from. Aborting the simulation!""")
                exit(1)

            nest.message(M_INFO, 'build_network', 'Connecting spike recorders.')
            nest.Connect(local_neurons[:self.__brunel_params['Nrec']], E_recorder,
                        'all_to_all')
            nest.Connect(E_stimulus, local_neurons[:self.__brunel_params['Nrec']],
                        'all_to_all')

        # read out time used for building
        BuildEdgeTime = time.time() - tic

        self.__logger.info(str(BuildEdgeTime) + ' # build_edge_time')
        self.__logger.info(str(memory_thisjob()) + ' # virt_mem_after_edges')

        return E_recorder if self.__sim_params['record_spikes'] else None

    def run_simulation(self, global_minimum_step_size):
        """Performs a simulation, including network construction"""

        nest.ResetKernel()
        nest.set_verbosity(M_INFO)

        self.__logger.info(str(memory_thisjob()) + ' # virt_mem_0')

        # sr = build_network(logger)

        tic = time.time()

        # nest.Simulate(params['presimtime'])

        # PreparationTime = time.time() - tic

        # self.__logger.info(str(memory_thisjob()) + ' # virt_mem_after_presim')
        # self.__logger.info(str(PreparationTime) + ' # presim_time')

        # tic = time.time()

        # nest.Simulate(params['simtime'])
        nest.Prepare()
        count = 0.0
        self.__logger.debug('starting simulation')
        # while count * self.__parameters.time_synch < self.__parameters.simulation_time:
        while count * global_minimum_step_size < 30:
            self.__logger.info(f"simulation run counter: {count+1}")
            nest.Run(1.5)
            count += 1

        SimCPUTime = time.time() - tic

        self.__logger.info(str(memory_thisjob()) + ' # virt_mem_after_sim')
        self.__logger.info(str(SimCPUTime) + ' # sim_time')

        # if self.__sim_params['record_spikes']:
        #     self.__logger.info(str(compute_rate(sr)) + ' # average rate')

        print(nest.kernel_status)


def compute_rate(sr):
    """Compute local approximation of average firing rate

    This approximation is based on the number of local nodes, number
    of local spikes and total time. Since this also considers devices,
    the actual firing rate is usually underestimated.

    """

    n_local_spikes = sr.n_events
    n_local_neurons = self.__brunel_params['Nrec']
    simtime = self.__sim_params['simtime']
    return 1. * n_local_spikes / (n_local_neurons * simtime) * 1e3


def memory_thisjob():
    """Wrapper to obtain current memory usage"""
    nest.ll_api.sr('memory_thisjob')
    return nest.ll_api.spp()


def lambertwm1(x):
    """Wrapper for LambertWm1 function"""
    # Using scipy to mimic the gsl_sf_lambert_Wm1 function.
    return sp.lambertw(x, k=-1 if x < 0 else 0).real
        
