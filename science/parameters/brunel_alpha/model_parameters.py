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

    Parameter section
    Define all relevant parameters: changes should be made here
"""
from dataclasses import dataclass
import numpy as np
import scipy.special as sp



@dataclass(frozen=True)
class Parameters:
    """
        Data class for defining the parameter values for Brunel Alpha Model
    """
    # 1. simulator specific parameters
    sim_parameters = {
    'nvp': 1,               # total number of virtual processes
    'scale': 1.,            # scaling factor of the network size
                            # total network size = scale*11250 neurons
    'simtime': 250.,        # total simulation time in ms
    'presimtime': 50.,      # simulation time until reaching equilibrium
    'dt': 0.1,              # simulation step
    'record_spikes': True,  # switch to record spikes of excitatory
                            # neurons to file
    'path_name': '.',       # path where all files will have to be written
    'log_file': 'log',      # naming scheme for the log files
    }

    # NOTE from example documentaion:
    #   For compatibility with earlier benchmarks, we require a rise time of
    #   ``t_rise = 1.700759 ms`` and we choose ``tau_syn`` to achieve this for given
    #   ``tau_m``. This requires numerical inversion of the expression for ``t_rise``
    #   in ``convert_synapse_weight``. We computed this value once and hard-code
    #   it here.

    # 2. tau_syn
    tau_syn = 0.32582722403722841

    # 3. Model Specific parameters
    brunel_params = {
        'NE': int(9000 * sim_parameters['scale']),  # number of excitatory neurons
        'NI': int(2250 * sim_parameters['scale']),  # number of inhibitory neurons

        'Nrec': 1000,  # number of neurons to record spikes from

        'model_params': {  # Set variables for iaf_psc_alpha
            'E_L': 0.0,  # Resting membrane potential(mV)
            'C_m': 250.0,  # Capacity of the membrane(pF)
            'tau_m': 10.0,  # Membrane time constant(ms)
            't_ref': 0.5,  # Duration of refractory period(ms)
            'V_th': 20.0,  # Threshold(mV)
            'V_reset': 0.0,  # Reset Potential(mV)
            # time const. postsynaptic excitatory currents(ms)
            'tau_syn_ex': tau_syn,
            # time const. postsynaptic inhibitory currents(ms)
            'tau_syn_in': tau_syn,
            'tau_minus': 30.0,  # time constant for STDP(depression)
            # V can be randomly initialized see below
            'V_m': 5.7  # mean value of membrane potential
        },

        # NOTE from example documentaion:
        #   Note that Kunkel et al. (2014) report different values. The values
        #   in the paper were used for the benchmarks on K, the values given
        #   here were used for the benchmark on JUQUEEN.

        'randomize_Vm': True,
        'mean_potential': 5.7,
        'sigma_potential': 7.2,

        'delay': 1.5,  # synaptic delay, all connections(ms)

        # synaptic weight
        'JE': 0.14,  # peak of EPSP

        'sigma_w': 3.47,  # standard dev. of E->E synapses(pA)
        'g': -5.0,

        'stdp_params': {
            'delay': 1.5,
            'alpha': 0.0513,
            'lambda': 0.1,  # STDP step size
            'mu': 0.4,  # STDP weight dependence exponent(potentiation)
            'tau_plus': 15.0,  # time constant for potentiation
        },

        'eta': 1.685,  # scaling of external stimulus
        'filestem': sim_parameters['path_name']
    }

    # @property
    # def sim_parameters(self): return self.__params

    # @property
    # def brunel_params(self): return self.__brunel_params

    # @property
    # def tau_syn(self): return self.__tau_syn


    def convert_synapse_weight(self, tau_m, tau_syn, C_m):
        """
        Computes conversion factor for synapse weight from mV to pA

        This function is specific to the leaky integrate-and-fire neuron
        model with alpha-shaped postsynaptic currents.
        """

        # compute time to maximum of V_m after spike input
        # to neuron at rest
        a = tau_m / tau_syn
        b = 1.0 / tau_syn - 1.0 / tau_m
        t_rise = 1.0 / b * (-self.lambertwm1(-np.exp(-1.0 / a) / a).real - 1.0 / a)

        v_max = np.exp(1.0) / (tau_syn * C_m * b) * (
            (np.exp(-t_rise / tau_m) - np.exp(-t_rise / tau_syn)) /
            b - t_rise * np.exp(-t_rise / tau_syn))
        return 1. / v_max

    def lambertwm1(self, x):
        """Wrapper for LambertWm1 function"""
        # Using scipy to mimic the gsl_sf_lambert_Wm1 function.
        return sp.lambertw(x, k=-1 if x < 0 else 0).real