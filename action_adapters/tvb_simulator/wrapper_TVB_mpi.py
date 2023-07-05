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
import sys
import numpy as np
from mpi4py import MPI

from EBRAINS_ConfigManager.global_configurations_manager.xml_parsers.default_directories_enum import DefaultDirectories
from EBRAINS_RichEndpoint.application_companion.common_enums import Response


class TVBMpiWrapper:
    def __init__(self, log_settings, configurations_manager, simulator_tvb,
                 intercalehub_nest_to_tvb=None,
                 intercalehub_tvb_to_nest=None) -> None:
        self.__logger = configurations_manager.load_log_configurations(
                name="TVB_MPI_Wrapper",
                log_configurations=log_settings,
                target_directory=DefaultDirectories.SIMULATION_RESULTS)

        self.__simulator_tvb = simulator_tvb
        # TODO rename variables
        self.__simulation_length = self.__simulator_tvb.simulation_length
        self.__dt = self.__simulator_tvb.integrator.dt
        self.__time_synch = self.__simulator_tvb.synchronization_time
        self.__time_synch_n = int(np.around(self.__time_synch / self.__dt))
        self.__nb_monitor = len(self.__simulator_tvb.monitors)
        self.__id_proxy = self.__simulator_tvb.proxy_inds
        # self.__interscalehub_address = interscalehub_address
        self.__intercalehub_nest_to_tvb = intercalehub_nest_to_tvb
        self.__intercalehub_tvb_to_nest = intercalehub_tvb_to_nest
        # receiver communicator
        self.__comm_receiver = []
        # sender communicator
        self.__comm_sender = []
        # initialise the variable for the saving the results
        self.__simulation_results = []
        for _ in range(self.__nb_monitor):  # the input output monitor
            self.__simulation_results.append([])

    def init_mpi(self):
        """sets up MPI communicators"""
        # create receiver communicator
        for _ in self.__id_proxy:
            self.__comm_receiver.append(
                self.__create_mpi_communicator(self.__intercalehub_nest_to_tvb))
        self.__logger.debug(f"receiver communicators: {self.__comm_receiver}")
        # create sender communicator
        for _ in self.__id_proxy:
            self.__comm_sender.append(
                self.__create_mpi_communicator(self.__intercalehub_tvb_to_nest))
        self.__logger.debug(f"sender communicators: {self.__comm_sender}")
        # TODO error handling

    def __create_mpi_communicator(self, interscalehub_address):
        """creates mpi Intercommunicators"""
        self.__logger.debug(f"connecting at {interscalehub_address}")
        comm = MPI.COMM_WORLD.Connect(interscalehub_address)
        self.__logger.info(f"connected to {interscalehub_address}")
        return comm

    def __send_mpi(self, comm, times, data):
        """
        send mpi data
        :param comm: MPI communicator
        :param times: times of values
        :param data: rates inputs
        :param logger: logger of the modules
        :return:nothing
        """
        self.__logger.info("start send")
        status_ = MPI.Status()
        # wait until the transformer accept the connections
        accept = False
        while not accept:
            req = comm.irecv(source=0, tag=0)
            accept = req.wait(status_)
            self.__logger.info("send accept")
        source = status_.Get_source()  # the id of the excepted source
        self.__logger.info("get source")
        data = np.ascontiguousarray(data, dtype='d')  # format the rate for sending
        shape = np.array(data.shape[0], dtype='i')  # size of data
        times = np.array(times, dtype='d')  # time of starting and ending step
        comm.Send([times, MPI.DOUBLE], dest=source, tag=0)
        comm.Send([shape, MPI.INT], dest=source, tag=0)
        comm.Send([data, MPI.DOUBLE], dest=source, tag=0)
        self.__logger.info("end send")

    def __mpi_receive(self, comm):
        """ 
            receive proxy values the
        :param comm: MPI communicator
        :param logger: logger of the modules
        :return: rate of all proxy
        """
        self.__logger.info("start receive")
        status_ = MPI.Status()
        # send to the transformer : I want the next part
        req = comm.isend(True, dest=0, tag=0)
        req.wait()
        time_step = np.empty(2, dtype='d')
        comm.Recv([time_step, 2, MPI.DOUBLE], source=0, tag=MPI.ANY_TAG, status=status_)
        # get the size of the rate
        size = np.empty(1, dtype='i')
        comm.Recv([size, MPI.INT], source=0, tag=0)
        # get the rate
        rates = np.empty(size, dtype='d')
        comm.Recv([rates, size, MPI.DOUBLE], source=0, tag=MPI.ANY_TAG, status=status_)
        self.__logger.info("end receive " + str(time_step))
        # print the summary of the data
        if status_.Get_tag() == 0:
            return time_step, rates
        else:
            return None

    def __end_mpi(self, comm, is_mode_sending):
        """
        ending the communication
        :param comm: MPI communicator
        :param path: for the close the port
        :param sending: if the transformer is for sending or receiving data
        :param logger: logger of the module
        :return: nothing
        """
        # different ending of the transformer
        if is_mode_sending:
            self.__logger.info(f"TVB close connection send {self.__intercalehub_tvb_to_nest}")
            sys.stdout.flush()
            status_ = MPI.Status()
            # wait until the transformer accept the connections
            self.__logger.info("TVB send check")
            accept = False
            while not accept:
                req = comm.irecv(source=0, tag=0)
                accept = req.wait(status_)
            self.__logger.info("TVB send end simulation")
            source = status_.Get_source()  # the id of the excepted source
            times = np.array([0., 0.], dtype='d')  # time of starting and ending step
            comm.Send([times, MPI.DOUBLE], dest=source, tag=1)
            self.__close_connection(comm, self.__intercalehub_tvb_to_nest)
        else:
            self.__logger.info("TVB close connection receive " + self.__intercalehub_nest_to_tvb)
            # send to the transformer : I want the next part
            req = comm.isend(True, dest=0, tag=1)
            req.wait()
            self.__close_connection(comm, self.__intercalehub_nest_to_tvb)
        # # closing the connection at this end
        # self.__logger.info("TVB disconnect communication")
        # comm.Disconnect()
        # self.__logger.info("TVB close " + self.__interscalehub_address)
        # MPI.Close_port(self.__interscalehub_address)
        # self.__logger.info("TVB close connection " + self.__interscalehub_address)
        return Response.OK

    def __close_connection(self, comm, address):
        # closing the connection at this end
        self.__logger.info("TVB disconnect communication")
        comm.Disconnect()
        self.__logger.info("TVB close " + address)
        MPI.Close_port(address)
        self.__logger.info("TVB close connection " + address)
    
    def __prepare_and_send_initialization_date(self):
        # prepare initialization data
        self.__logger.info("send initialization of TVB: prepare data")
        initialization_data = []
        for i in np.arange(0, self.__time_synch_n, 1, dtype=np.int):
            initialization_data.append(self.__simulator_tvb._loop_compute_node_coupling(i)[:, self.__id_proxy, :])
        initialization_data = np.concatenate(initialization_data)
        time_init = [0, self.__time_synch]

        # send initialization data
        self.__logger.info("send initialization of TVB: send data")
        for index, comm in enumerate(self.__comm_sender):
            self.__send_mpi(comm, time_init, initialization_data[:, index] * 1e3)

    def __receive_data(self):
        """
        helper function to receive data (spikes) from
        InterscaleHub_NEST_to_TVB using MPI
        """
        data_value = []
        self.__logger.debug("start receiving data")
        for comm in self.__comm_receiver:
            receive = self.__mpi_receive(comm)
            time_data = receive[0]
            data_value.append(receive[1])
        self.__logger.debug(f"time received: {time_data}, data received: {data_value}")
        return data_value, time_data, receive  # spikes

    def __format_and_reshape_simulation_data(self, data_value, time_data, receive):
        """helper function to format and reshape simulation data"""
        data = np.empty((2,), dtype=object)
        nb_step = np.rint((time_data[1] - time_data[0]) / self.__dt)
        nb_step_0 = np.rint(time_data[0] / self.__dt) + 1  # start at the first time step not at 0.0
        time_data = np.arange(nb_step_0, nb_step_0 + nb_step, 1) * self.__dt
        data_value = np.swapaxes(np.array(data_value), 0, 1)[:, :]
        data_value = np.expand_dims(data_value, axis=(1, 3))
        # check time and data shapes
        if data_value.shape[0] != time_data.shape[0]:
            self.__logger.critical(nb_step)
            self.__logger.critical(self.__dt, time_data, receive)
            self.__logger.critical(f"Bad shape of data:{data_value.shape[0]}, time shape: {time_data.shape[0]}")
            # TODO handle exception
            raise (Exception('Bad shape of data ' + str(data_value.shape[0]) + " " + str(time_data.shape[0])))
        
        # all is fine
        self.__logger.debug(f"after formatting, time:{time_data}, data:{data_value}")
        data[:] = [time_data, data_value]
        return data
    
    def __run_tvb_simulation(self, data):
        """helper function to run TVB simulation with updated data"""
        self.__logger.info("TVB start simulation "
                           f"{self.__simulation_run_counter * self.__time_synch}")
        # start simulation until next synchronization time check
        for result in self.__simulator_tvb(simulation_length=self.__time_synch, cosim_updates=data):
            for i in range(self.__nb_monitor):
                if result[i] is not None:
                    # save results of current simulation run
                    self.__simulation_results[i].append(result[i])
        self.__logger.info(" TVB end simulation")
    
    def __send_data(self):
        """
        helper function to send data (rates) to InterscaleHub_TVB_to_NEST using MPI
        """
        # get TVB output (rates) for NEST
        data_for_nest = self.__simulator_tvb.loop_cosim_monitor_output(n_steps=self.__time_synch_n)[0]
        times = [data_for_nest[0][0], data_for_nest[0][-1]]
        rate = np.concatenate(data_for_nest[1][:, 0, [self.__id_proxy], 0])
        for index, comm in enumerate(self.__comm_sender):
            self.__send_mpi(comm, times, rate[:, index] * 1e3)
        self.__logger.debug("data is send")

    def __finalize(self):
        """helper function to end communications and finalize MPI"""
        # close ports and send signal to end communications by
        # Inter-communicator for sending MPI data
        for index, comm in enumerate(self.__comm_sender):
            self.__logger.info('end comm send')
            self.__end_mpi(comm, is_mode_sending=True)
        
        # close ports and send signal to end communications by
        # Inter-communicator for receiving MPI data
        for index, comm in enumerate(self.__comm_receiver):
            self.__logger.info('end comm receive')
            self.__end_mpi(comm, is_mode_sending=False)
        
        # ending with MPI
        MPI.Finalize()

    def __reshape_result(self, result):
        """reshapes the output of TVB for the"""
        times = []
        values = []
        for (running_time, running_value) in result[0]:
            if running_time > 0.0:
                times.append(running_time)
                values.append(running_value)
        return ([np.array(times), np.expand_dims(np.concatenate(values), 1)],)
    
    def run_simulation_and_data_exchange(self, global_minimum_step_size):
        """
        return the result of the simulation between the wanted time
        :param simulator: tvb simulator
        :param path: the folder of the simulation
        :param logger: logger of the run
        """
        # prepare and send initialization data, required by protocol to signal
        # ready to receive
        self.__prepare_and_send_initialization_date()
        self.__simulation_run_counter = 0
        # the main loop of the simulation and data exchange
        # while self.__simulation_run_counter * self.__time_synch < self.__simulation_length:
        while self.__simulation_run_counter * global_minimum_step_size < self.__simulation_length:
            # 1. receive data from InterscaleHub_NEST_to_TVB
            data_value, time_data, receive = self.__receive_data()
            # 2. format time and data for input to TVB simulation
            data = self.__format_and_reshape_simulation_data(data_value, time_data, receive)
            # 3. run TVB simulation until next synchronization time check with
            # data received from NEST
            self.__run_tvb_simulation(data)
            # 4. send data to InterscaleHub_TVB_to_NEST
            self.__send_data()
            # 5. increment of the loop
            self.__simulation_run_counter += 1
            # 6. continue simulation and data exchange
            continue

        # finishes simulation and data exchange
        # now save the last part
        self.__logger.info(" TVB finish")
        # end communications and finalize MPI
        self.__finalize()
        self.__logger.info(" TVB exit")
        return self.__reshape_result(self.__simulation_results)
