from __future__ import print_function
from __future__ import division
from future import standard_library
standard_library.install_aliases()
#from builtins import str
from builtins import range
from quantities.quantity import Quantity
import sciunit
from sciunit import Test,Score
try:
    from sciunit import ObservationError
except:
    from sciunit.errors import ObservationError
import hippounit.capabilities as cap
from sciunit.utils import assert_dimensionless# Converters.
from sciunit.scores import BooleanScore,ZScore # Scores.

try:
    import numpy
except:
    print("NumPy not loaded.")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#from neuron import h
import collections
import efel
import os
import multiprocessing
import multiprocessing.pool
import functools
import math
from scipy import stats

from scipy.optimize import fsolve, curve_fit
import scipy.interpolate as interpolate

import json
from hippounit import plottools
import collections


try:
    import pickle as pickle
except:
    import pickle
import gzip

try:
    import copy_reg
except:
    import copyreg

from types import MethodType

from quantities import mV, nA, ms, V, s

from hippounit import scores



def _pickle_method(method):
    func_name = method.__func__.__name__
    obj = method.__self__
    cls = method.__self__.__class__
    return _unpickle_method, (func_name, obj, cls)

def _unpickle_method(func_name, obj, cls):
    for cls in cls.mro():
        try:
            func = cls.__dict__[func_name]
        except KeyError:
            pass
        else:
            break
    return func.__get__(obj, cls)


# https://stackoverflow.com/questions/6974695/python-process-pool-non-daemonic
class NoDaemonProcess(multiprocessing.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

class NoDeamonPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess

try:
    copy_reg.pickle(MethodType, _pickle_method, _unpickle_method)
except:
    copyreg.pickle(MethodType, _pickle_method, _unpickle_method)


class PathwayInteraction(Test):
    """Tests the signal integration in oblique dendrites for increasing number of synchronous and asynchronous inputs"""

    def __init__(self, config = {},
                 observation = {}  ,
                 name="Pathway Interaction test" ,
                 force_run_adjust_syn_weight=False,
                 base_directory= None,
                 num_of_dend_locations = 15,
                 random_seed = 1,
                 show_plot=True,
                 save_all = True,
                 AMPA_weight_init = 0.000748):

        self.num_of_dend_locations = num_of_dend_locations
        self.random_seed = random_seed

        # observation = self.format_data(observation)
        # observation = self.add_std_to_observation(observation)

        Test.__init__(self, observation, name)

        self.required_capabilities = (cap.ProvidesRandomDendriticLocations, cap.ProvidesRecordingLocationsOnTrunk, cap.ReceivesSynapse, cap.InitialiseModel, cap.ThetaSynapticStimuli, cap.RunSimulation_ReturnTraces, cap.NumOfPossibleLocations) # +=

        self.force_run_adjust_syn_weight = force_run_adjust_syn_weight
        self.show_plot = show_plot
        self.save_all = save_all

        self.base_directory = base_directory

        self.path_figs = None    #added later, because model name is needed
        self.path_results = None

        self.logFile = None
        self.test_log_filename = 'test_log.txt'

        self.config = config

        self.npool = multiprocessing.cpu_count() - 1

        self.AMPA_weight_init = AMPA_weight_init

        description = ""

    score_type = scores.ZScore_ObliqueIntegration

    def format_data(self, observation):

        for key, val in list(observation.items()):
            try:
                assert type(observation[key]) is Quantity
            except Exception as e:
                try:
                    observation[key] = float(val)
                except Exception as e:
                    quantity_parts = val.split(" ")
                    number = float(quantity_parts[0])
                    units = " ".join(quantity_parts[1:])
                    observation[key] = Quantity(number, units)
        return observation


    def analyse_syn_traces(self, model, t, v, t_no_input, v_no_input):

        if not numpy.array_equal(t, t_no_input):    #if the  time vectors are not equal, the traces are resampled with fixed time step
            dt = 0.025
            time_vector = numpy.arange(t[0], t[-1], dt)  #from the first to the last element of the original time vector

            interp_trace = numpy.interp(time_vector, t, v)
            interp_trace_no_input = numpy.interp(time_vector, t, v_no_input)

            depol = interp_trace - interp_trace_no_input


            print("Voltage traces are resampled using linear interpolation")

        else:
            depol = v - v_no_input
            time_vector = t

        max_depol = max(depol)

        return max_depol  


    def synapse(self, model, t_no_input, v_no_input, weight, path_adjust_syn_weight, dend_loc0):
        file_name = path_adjust_syn_weight + 'Trace_' + str(dend_loc0[0]) + '(' + str(dend_loc0[1]) + ')_' + 'weight_' + str(weight) + '.p'

        if self.force_run_adjust_syn_weight or (os.path.isfile(file_name) is False):

            t, v, v_dend = model.run_synapse_get_vm(dend_loc0, weight)
            if self.save_all:
                pickle.dump([t, v, v_dend], gzip.GzipFile(file_name, "wb"))

        else:
            [t, v, v_dend] = pickle.load(gzip.GzipFile(file_name, "rb"))

        max_soma_depol = self.analyse_syn_traces(model, t, v, t_no_input, v_no_input)

        return max_soma_depol 

    def adjust_syn_weight(self, model, dend_loc, pathway):

        if self.base_directory:
            path_adjust_syn_weight = self.base_directory + 'temp_data/' + 'pathway_interaction/adjust_syn_weight/' + model.name + '/'
        else:
            path_adjust_syn_weight = model.base_directory + 'temp_data/' + 'pathway_interaction/adjust_syn_weight/'

        try:
            if not os.path.exists(path_adjust_syn_weight) and self.save_all:
                os.makedirs(path_adjust_syn_weight)
        except OSError as e:
            if e.errno != 17:
                raise
            pass

        if pathway == 'SC':
            file_name = path_adjust_syn_weight + 'SC_weight.p'
            desired_somatic_depol = 0.2
        if pathway == 'PP':
            file_name = path_adjust_syn_weight + 'PP_weight.p'
            desired_somatic_depol = 0.2
        file_name_no_input = path_adjust_syn_weight + 'SomaticTrace_no_input.p'

        # SC_desired_somatic_depol = 0.2
        # PP_desired_somatic_depol = 0.2

        if self.force_run_adjust_syn_weight or (os.path.isfile(file_name_no_input) is False):


            pool_syn_ = multiprocessing.Pool(1, maxtasksperchild = 1)    # I use multiprocessing to keep every NEURON related task in independent processes
            t_no_input, v_no_input, v_dend_no_input = pool_syn_.apply(model.run_synapse_get_vm, args = (dend_loc[0], 0.0))
            # plt.plot(t_no_input, v_no_input)
            # plt.show()
            pool_syn_.terminate()
            pool_syn_.join()
            del pool_syn_

            if self.save_all:
                pickle.dump([t_no_input, v_no_input, v_dend_no_input], gzip.GzipFile(file_name_no_input, "wb"))

        else:
            [t_no_input, v_no_input, v_dend_no_input] = pickle.load(gzip.GzipFile(file_name_no_input, "rb"))


        synapse_ = functools.partial(self.synapse, model, t_no_input, v_no_input, self.AMPA_weight_init, path_adjust_syn_weight)

        pool_syn = multiprocessing.Pool(self.npool, maxtasksperchild = 1)    # I use multiprocessing to keep every NEURON related task in independent processes
        max_soma_depols = pool_syn.map(synapse_, dend_loc, chunksize=1)
        pool_syn.terminate()
        pool_syn.join()
        del pool_syn

        print("before:" , max_soma_depols)
        avg_max_soma_depols = numpy.mean(max_soma_depols)
        print('avg before', avg_max_soma_depols)

        scale_factor = desired_somatic_depol / avg_max_soma_depols
        print('scale_factor', scale_factor)

        synapse_ = functools.partial(self.synapse, model, t_no_input, v_no_input, self.AMPA_weight_init * scale_factor, path_adjust_syn_weight)

        pool_syn = multiprocessing.Pool(self.npool, maxtasksperchild = 1)    # I use multiprocessing to keep every NEURON related task in independent processes
        max_soma_depols = pool_syn.map(synapse_, dend_loc, chunksize=1)
        pool_syn.terminate()
        pool_syn.join()
        del pool_syn

        print("after:" , max_soma_depols)
        avg_max_soma_depols = numpy.mean(max_soma_depols)
        print('avg after', avg_max_soma_depols)

        AMPA_weight_final = self.AMPA_weight_init * scale_factor

        pickle.dump(AMPA_weight_final, gzip.GzipFile(file_name, "wb"))

        return AMPA_weight_final

    def adjust_num_syn(self, model, SC_weight, PP_weight, recording_loc, stimuli_params, t_no_input_rec_dend, v_no_input_rec_dend, pathway):
        interval_bw_trains, interval_bw_stimuli_in_train, num_trains, num_stimuli_in_train = stimuli_params

        new_stimuli_params = [interval_bw_trains, interval_bw_stimuli_in_train, 1, num_stimuli_in_train] 

        dist_range = [0,9999999999]
        random_seed = self.random_seed

        if pathway == 'SC':

            model.SecList = model.ObliqueSecList_name
            dend_loc, locations_distances = model.get_random_locations_multiproc(10, self.random_seed, dist_range) # number of random locations , seed
            PP_dend_loc =[] 
            num_of_loc = model.get_num_of_possible_locations()
            
            exp_depol = 16.0
            exp_depol_sd = 1.6

            # traces = self.theta_pathway_stimulus(model, SC_weight, PP_weight, dend_loc, PP_dend_loc, recording_loc, new_stimuli_params, 600, pathway)


        elif pathway == 'PP':
            model.SecList = model.TuftSecList_name
            dend_loc, locations_distances = model.get_random_locations_multiproc(10, self.random_seed, dist_range) # number of random locations , seed
            
            SC_dend_loc =[] 
            num_of_loc = model.get_num_of_possible_locations()

            exp_depol = 10.2
            exp_depol_sd = 1.0

            # traces = self.theta_pathway_stimulus(model, SC_weight, PP_weight, SC_dend_loc, dend_loc, recording_loc, new_stimuli_params, 600, pathway)


        # max_depol = self.analyse_syn_traces(model, traces[pathway]['t'], traces[pathway]['v_dend'], t_no_input_rec_dend, v_no_input_rec_dend)
        # plt.figure()
        # plt.plot(traces[pathway]['t'], traces[pathway]['v_dend'])
        # plt.show()
        # print(max_depol)

        found = False
        prev_max_depol = None

        print(pathway, 'num_of_loc', num_of_loc)

        while not found and len(dend_loc) > 1 and len(dend_loc) <= num_of_loc:

            random_seed += 1

            if prev_max_depol:
                prev_max_depol = max_depol    # if it already has a value (we are not in the first iteration), it gets the value of the previous iteration 

            pool = multiprocessing.Pool(1, maxtasksperchild = 1)   # multiprocessing pool is used so that the model can be killed after the simulation, avoiding pickle errors
        
            if pathway == 'SC':
                traces = pool.apply(self.theta_pathway_stimulus, args = (model, SC_weight, PP_weight, dend_loc, PP_dend_loc, recording_loc, new_stimuli_params, 600, pathway))

            elif pathway == 'PP':
                traces = pool.apply(self.theta_pathway_stimulus, args = (model, SC_weight, PP_weight, SC_dend_loc, dend_loc, recording_loc, new_stimuli_params, 600, pathway))

            pool.terminate()
            pool.join()
            del pool


            max_depol = self.analyse_syn_traces(model, traces[pathway]['t'], traces[pathway]['v_dend'], t_no_input_rec_dend, v_no_input_rec_dend)
            print(pathway, ': ', max_depol)


            if not prev_max_depol:         # if it has a value of None (we are in the first iteration), it gets the same value as the max_depol 
                prev_max_depol = max_depol


            if max_depol < exp_depol - exp_depol_sd and prev_max_depol < exp_depol - exp_depol_sd:
                if pathway == 'SC':
                    model.SecList = model.ObliqueSecList_name
                elif pathway == 'PP':
                    model.SecList = model.TuftSecList_name
                
                prev_dend_loc = list(dend_loc)
     
                dend_loc_, locations_distances_ = model.get_random_locations_multiproc(1, random_seed, dist_range) # select one more location

                while dend_loc_[0] in dend_loc and len(dend_loc) <= num_of_loc: 
                    random_seed += 1
                    dend_loc_, locations_distances_ = model.get_random_locations_multiproc(1, random_seed, dist_range) # select one more location
                dend_loc.append(dend_loc_[0]) 
                print(pathway, ': ', dend_loc)

            elif max_depol < exp_depol - exp_depol_sd and prev_max_depol > exp_depol + exp_depol_sd:
                print(pathway, ' koztes1')
                
                print('depols', max_depol, prev_max_depol)
                print('dend_locs')
                print(dend_loc)
                print(prev_dend_loc)
                accepted_depol_diff= min(abs(max_depol-exp_depol), abs(prev_max_depol-exp_depol))
                if accepted_depol_diff == abs(prev_max_depol-exp_depol):
                    dend_loc = prev_dend_loc
                    found = True
                else:
                    # dend_loc remains
                    found = True 
                print('chosen dend_loc', dend_loc)

            elif max_depol > exp_depol + exp_depol_sd and prev_max_depol > exp_depol + exp_depol_sd:
                prev_dend_loc = list(dend_loc)

                dend_loc.pop()  #removing last element
                print(pathway, ': ', dend_loc)

            elif max_depol > exp_depol + exp_depol_sd and prev_max_depol < exp_depol - exp_depol_sd:
                print(pathway, ' koztes2')

                print('depols', max_depol, prev_max_depol)
                print('dend_locs')
                print(dend_loc)
                print(prev_dend_loc)

                accepted_depol_diff= min(abs(max_depol-exp_depol), abs(prev_max_depol-exp_depol))
                if accepted_depol_diff == abs(prev_max_depol-exp_depol):
                    dend_loc = list(prev_dend_loc)
                    found = True
                else:
                    # dend_loc remains
                    found = True 
                print('chosen dend_loc', dend_loc)

            elif exp_depol - exp_depol_sd < max_depol < exp_depol + exp_depol_sd:

                found = True
                print(pathway, ': ', dend_loc)



        if not found:
            print("The number of activated synapses could not be adjusted properly on pathway:", pathway)
            print("Maximum depolarization achieved:", max_depol, "mV")
            print("Stimulated dendritic locations:", dend_loc)

        return {pathway: dend_loc}    


    def generate_no_input_dend_trace(self, model, recording_loc):
        model.initialise()
        t, v, v_dend = model.run_simulation([], recording_loc, 600)
        return t, v, v_dend
        
    def theta_pathway_stimulus(self, model, SC_weight, PP_weight, SC_dend_loc, PP_dend_loc, recording_loc, stimuli_params, tstop, pathway):

        """Simulates pathway stimulation of the Schaffer-collateral or the Perforant Path, or both at the same time. The simultaneous activation of the 2 pathways is solved by calling the same Capability function but with different arguments (section list, synaptic parameters). For this to be feasible, the model must be loaded first, and therefore separate capability methods are needed to (1) load the model, (2) define the synaptic stimulus and (3) to run the simulation and make the recordings. In other tests all of these were done through a single capability method."""

        '''
        interval_bw_trains = 1/ self.config["frequency of stimulus sequence"] * 1000
        interval_bw_stimuli_in_train = 1/ self.config["frequency of trains"] * 1000
        num_trains = self.config["number of trains"]
        num_stimuli_in_train = self.config["number of stimuli in a train"]
        '''

        interval_bw_trains, interval_bw_stimuli_in_train, num_trains, num_stimuli_in_train = stimuli_params

        model.initialise()  # should be solved more like using Capabilities (problem: to add synapses, model should be loaded, but can not be reloaded. We want to be able to add PP and SC stimulation separately. (Later different synaptic parameters, different delay etc) 

        if pathway == 'SC':
            model.activate_theta_stimuli(SC_dend_loc, SC_weight, pathway, interval_bw_trains, interval_bw_stimuli_in_train, num_trains, num_stimuli_in_train)
            t, v, v_dend = model.run_simulation(SC_dend_loc, recording_loc, tstop)
            '''
            plt.figure()
            plt.plot(t,v)
            plt.plot(t,v_dend)
            plt.title('SC stimulus')
            '''
        elif pathway == 'PP':
            model.activate_theta_stimuli(PP_dend_loc, PP_weight, pathway, interval_bw_trains, interval_bw_stimuli_in_train, num_trains, num_stimuli_in_train)
            t, v, v_dend = model.run_simulation(PP_dend_loc, recording_loc, tstop)
            '''
            plt.figure()
            plt.plot(t,v)
            plt.plot(t,v_dend)
            plt.title('PP stimulus')
            '''
        elif pathway == 'SC+PP':
            model.activate_theta_stimuli(PP_dend_loc, PP_weight, 'PP', interval_bw_trains, interval_bw_stimuli_in_train, num_trains, num_stimuli_in_train)
            # model.activate_theta_stimuli(SC_dend_loc + PP_dend_loc, PP_weight, 'PP')            
            model.activate_theta_stimuli(SC_dend_loc, SC_weight, 'SC', interval_bw_trains, interval_bw_stimuli_in_train, num_trains, num_stimuli_in_train)
            t, v, v_dend = model.run_simulation(SC_dend_loc + PP_dend_loc, recording_loc, tstop)
            '''
            plt.figure()
            plt.plot(t,v)
            plt.plot(t,v_dend)
            plt.title('SC+PP stimulus')
            '''
        # plt.show()
        traces = {pathway: {'t' : t, 'v_soma' : v, 'v_dend' : v_dend}} 

        return traces

    def plot_traces(self, traces_dict):
        
        fig= plt.figure()
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)

        ax1.plot(traces_dict['SC']['t'], traces_dict['SC']['v_soma'], label = 'soma')
        ax1.plot(traces_dict['SC']['t'], traces_dict['SC']['v_dend'], label = 'distal dendrite')
        ax1.set_xlabel('Time (ms)')
        ax1.set_ylabel('Voltage (mV)')
        ax1.title.set_text('SC stimulus')

        ax2.plot(traces_dict['PP']['t'], traces_dict['PP']['v_soma'])
        ax2.plot(traces_dict['PP']['t'], traces_dict['PP']['v_dend'])
        ax2.set_xlabel('Time (ms)')
        ax2.set_ylabel('Voltage (mV)')
        ax2.title.set_text('PP stimulus')

        ax3.plot(traces_dict['SC+PP']['t'], traces_dict['SC+PP']['v_soma'])
        ax3.plot(traces_dict['SC+PP']['t'], traces_dict['SC+PP']['v_dend'])
        ax3.set_xlabel('Time (ms)')
        ax3.set_ylabel('Voltage (mV)')
        ax3.title.set_text('SC+PP stimulus')
  
        fig.subplots_adjust(wspace = 0.5, hspace = 0.6)
        handles, labels = ax1.get_legend_handles_labels()
        fig.legend(handles, labels, bbox_to_anchor=(1.0, 1.0), loc = 'upper left')

        plt.figure()
        plt.plot(traces_dict['SC']['t'], traces_dict['SC']['v_soma'], label = 'soma')
        plt.plot(traces_dict['SC']['t'], traces_dict['SC']['v_dend'], label = 'distal dendrite')
        plt.xlabel('Time (ms)')
        plt.ylabel('Voltage (mV)')
        plt.title('SC stimulus')
        plt.legend(bbox_to_anchor=(1.0, 1.0), loc = 'upper left')

        plt.figure()
        plt.plot(traces_dict['PP']['t'], traces_dict['PP']['v_soma'], label = 'soma')
        plt.plot(traces_dict['PP']['t'], traces_dict['PP']['v_dend'], label = 'distal dendrite')
        plt.xlabel('Time (ms)')
        plt.ylabel('Voltage (mV)')
        plt.title('PP stimulus')
        plt.legend(bbox_to_anchor=(1.0, 1.0), loc = 'upper left')

        plt.figure()
        plt.plot(traces_dict['SC+PP']['t'], traces_dict['SC+PP']['v_soma'], label = 'soma')
        plt.plot(traces_dict['SC+PP']['t'], traces_dict['SC+PP']['v_dend'], label = 'distal dendrite')
        plt.xlabel('Time (ms)')
        plt.ylabel('Voltage (mV)')
        plt.title('SC+PP stimulus')
        plt.legend(bbox_to_anchor=(1.0, 1.0), loc = 'upper left')



    def validate_observation(self, observation):

        #self.add_std_to_observation(observation)

        try:
            pass

        except Exception as e:
            raise ObservationError(("Observation must be of the form "
                                    "{'mean':float*mV,'std':float*mV}"))


    def generate_prediction(self, model, verbose=False):
        """Implementation of sciunit.Test.generate_prediction."""

        efel.reset()
        plt.close('all') #needed to avoid overlapping of saved images when the test is run on multiple models in a for loop

        if self.base_directory:
            self.path_results = self.base_directory + 'results/' + 'pathway_interaction/' + model.name + '/'
        else:
            self.path_results = model.base_directory + 'results/' + 'pathway_interaction/'

        try:
            if not os.path.exists(self.path_results):
                os.makedirs(self.path_results)
        except OSError as e:
            if e.errno != 17:
                raise
            pass

        dist_range = [0,9999999999]

        model.SecList = model.ObliqueSecList_name
        SC_dend_loc, SC_locations_distances = model.get_random_locations_multiproc(self.num_of_dend_locations, self.random_seed, dist_range) # number of random locations , seed

        model.SecList = model.TuftSecList_name
        PP_dend_loc, PP_locations_distances = model.get_random_locations_multiproc(self.num_of_dend_locations, self.random_seed, dist_range) # number of random locations , seed

        """Finding recording location on Trunk whose distance is closest to 300 um"""
        distances = [self.config["distance of recording location"]]
        tolerance = self.config["distance tolerance"]

        rec_locs, rec_locs_actual_distances = model.find_trunk_locations_multiproc(distances, tolerance)
        print("recording locs", rec_locs, rec_locs_actual_distances)

        # recording_loc = min(rec_locs_actual_distances, key=abs(distances[0] - rec_locs_actual_distances.get))
        recording_loc = min(rec_locs_actual_distances.items(), key=lambda kv : abs(kv[1] - distances[0]))
        print(recording_loc, type(recording_loc))


        if not model.AMPA_name:
            print('')
            print('The built in Exp2Syn is used as the AMPA component. Tau1 =', model.AMPA_tau1, ',Tau2 =', model.AMPA_tau2 , '.')
            print('')
        if not model.NMDA_name: 
            print('')
            print('The default NMDA model of HippoUnit is used with Jahr, Stevens voltage dependence.')
            print('')

        print("Adjusting synaptic weights ...")

        SC_weight = self.adjust_syn_weight(model, SC_dend_loc, pathway = 'SC')

        PP_weight = self.adjust_syn_weight(model, PP_dend_loc, pathway = 'PP')
        

        pool = multiprocessing.Pool(1, maxtasksperchild = 1)
        t_no_input_rec_dend, v_soma_no_input, v_no_input_rec_dend  = pool.apply(self.generate_no_input_dend_trace, (model, recording_loc,)) # this is run in multiprocessing pool so that the model can be completely killed after done 
        pool.terminate()
        pool.join()
        del pool


        interval_bw_trains = 1/ self.config["frequency of stimulus sequence"] * 1000
        interval_bw_stimuli_in_train = 1/ self.config["frequency of trains"] * 1000
        num_trains = self.config["number of trains"]
        num_stimuli_in_train = self.config["number of stimuli in a train"]


        stimuli_params =[interval_bw_trains, interval_bw_stimuli_in_train, num_trains, num_stimuli_in_train] 

        # self.adjust_num_syn(model, SC_weight, PP_weight, recording_loc, stimuli_params, t_no_input_rec_dend, v_no_input_rec_dend, 'SC')

        pool = NoDeamonPool(self.npool, maxtasksperchild=1)  # NoDeamonPool is needed because Random locations are needed to be chosen several times, for which the model is loaded in a multiprocessing pool 
        adjust_num_syn_= functools.partial(self.adjust_num_syn, model, SC_weight, PP_weight, recording_loc, stimuli_params, t_no_input_rec_dend, v_no_input_rec_dend)
        dend_locs = pool.map(adjust_num_syn_, ['SC', 'PP'], chunksize=1)

        pool.terminate()
        pool.join()
        del pool

        dend_locs_dict = {} 
        for locs in dend_locs:
            dend_locs_dict.update(locs)

        
        SC_dend_loc = dend_locs_dict['SC'] 
        PP_dend_loc = dend_locs_dict['PP']

        tstop = 1600

        pool = multiprocessing.Pool(self.npool, maxtasksperchild=1)
        theta_pathway_stimulus_= functools.partial(self.theta_pathway_stimulus, model, SC_weight, PP_weight, SC_dend_loc, PP_dend_loc, recording_loc, stimuli_params, tstop)
        traces = pool.map(theta_pathway_stimulus_, ['SC', 'PP', 'SC+PP'], chunksize=1)

        pool.terminate()
        pool.join()
        del pool

        traces_dict = {} 
        for trace in traces:
            traces_dict.update(trace)
        # print(traces_dict)


        self.plot_traces(traces_dict)




   
        plt.show()


        ''' printing to logFile'''
        filepath = self.path_results + self.test_log_filename
        self.logFile = open(filepath, 'w') # if it is opened before multiprocessing, the multiporeccing won't work under python3

        self.logFile.write('Dendrites and locations to be tested:\n'+ str(dend_loc)+'\n')
        self.logFile.write("---------------------------------------------------------------------------------------------------\n")

        if not model.AMPA_name:
            self.logFile.write('The built in Exp2Syn is used as the AMPA component. Tau1 = ' + str(model.AMPA_tau1) + ', Tau2 = ' + str(model.AMPA_tau2) + '.\n')
            self.logFile.write("---------------------------------------------------------------------------------------------------\n")
        if not model.NMDA_name:
            self.logFile.write('The default NMDA model of HippoUnit is used with Jahr, Stevens voltage dependence.\n')
            self.logFile.write("---------------------------------------------------------------------------------------------------\n")

        for i in range(0, len(dend_loc)):

                if results0[i][0]=='always too small':
                    self.logFile.write('The somatic EPSP amplitude to input at ' + str(dend_loc[i][0]) +'('+str(dend_loc[i][1])+')' + ' was always lower than the desired value\n')
                    self.logFile.write("---------------------------------------------------------------------------------------------------\n")

                elif results0[i][0]=='always too big':
                    self.logFile.write('The somatic EPSP amplitude to input at ' + str(dend_loc[i][0]) +'('+str(dend_loc[i][1])+')' +  ' was always higher than the desired value\n')
                    self.logFile.write("---------------------------------------------------------------------------------------------------\n")



        print("Results are saved in the directory: ", self.path_results)

        efel.reset()

        return prediction

    def compute_score(self, observation, prediction, verbose=False):
        """Implementation of sciunit.Test.score_prediction."""


        file_name = self.path_results + 'oblique_features.p'



        return score

    def bind_score(self, score, model, observation, prediction):


        return score
