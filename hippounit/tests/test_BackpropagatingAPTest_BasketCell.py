from __future__ import print_function
from __future__ import division
from future import standard_library
standard_library.install_aliases()
#from builtins import str
from builtins import range
from quantities.quantity import Quantity
from quantities import mV, nA
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


try:
    copy_reg.pickle(MethodType, _pickle_method, _unpickle_method)
except:
    copyreg.pickle(MethodType, _pickle_method, _unpickle_method)


class BackpropagatingAPTest_BasketCell(Test):
    """Tests efficacy  and shape of back-propagating action potentials on the basal and apical dendrites of basket cells."""

    def __init__(self, config = {},
                observation = {},
                name="Back-propagating action potential test" ,
                force_run=False,
                base_directory= None,
                show_plot=True,
                save_all = True,
                num_of_apical_dend_locations = 15,
                num_of_basal_dend_locations = 15,
                random_seed = 1):

        observation = self.format_data(observation)

        Test.__init__(self, observation, name)

        self.required_capabilities += (cap.ReceivesSquareCurrent_ProvidesResponse_MultipleLocations,
                                        cap.ProvidesRandomDendriticLocations, cap.ReceivesSquareCurrent_ProvidesResponse,)

        self.force_run = force_run
        self.random_seed = random_seed

        self.show_plot = show_plot
        self.save_all = save_all

        self.base_directory = base_directory
        self.path_temp_data = None #added later, because model name is needed
        self.path_figs = None
        self.path_results = None

        self.logFile = None
        self.test_log_filename = 'test_log.txt'

        self.npool = multiprocessing.cpu_count() - 1

        self.num_of_apical_dend_locations = num_of_apical_dend_locations
        self.num_of_basal_dend_locations = num_of_basal_dend_locations

        self.config = config

        description = "Tests efficacy  and shape of back-propagating action potentials on the basal and apical dendrites of basket cells."

    score_type = scores.ZScore_backpropagatingAP

    def format_data(self, observation):
        for key, val in list(observation.items()):
            if key == 'soma':
                for ke, va in list(observation[key].items()): 
                    try:
                        assert type(observation[key][ke]['mean']) is Quantity
                        assert type(observation[key][ke]['std']) is Quantity
                    except Exception as e:
                        quantity_parts = va['mean'].split(" ")
                        number = float(quantity_parts[0])
                        units = " ".join(quantity_parts[1:])
                        observation[key][ke]['mean'] = Quantity(number, units)

                        quantity_parts = va['std'].split(" ")
                        number = float(quantity_parts[0])
                        units = " ".join(quantity_parts[1:])
                        observation[key][ke]['std'] = Quantity(number, units)
            else: 
                for ke, va in list(observation[key].items()): 
                    for k, v in list(observation[key][ke].items()): 
                        try:
                            assert type(observation[key][ke][k]['mean']) is Quantity
                            assert type(observation[key][ke][k]['std']) is Quantity
                        except Exception as e:
                            quantity_parts = v['mean'].split(" ")
                            number = float(quantity_parts[0])
                            units = " ".join(quantity_parts[1:])
                            observation[key][ke][k]['mean'] = Quantity(number, units)

                            quantity_parts = v['std'].split(" ")
                            number = float(quantity_parts[0])
                            units = " ".join(quantity_parts[1:])
                            observation[key][ke][k]['std'] = Quantity(number, units)
        # print(observation)
        return observation


    def cclamp(self, model, amp, delay, dur, section_stim, loc_stim, dend_locations):

        if self.base_directory:
            self.path_temp_data = self.base_directory + 'temp_data/' + 'backpropagating_AP_BC/' + model.name + '/'
        else:
            self.path_temp_data = model.base_directory + 'temp_data/' + 'backpropagating_AP_BC/'


        try:
            if not os.path.exists(self.path_temp_data) and self.save_all:
                os.makedirs(self.path_temp_data)
        except OSError as e:
            if e.errno != 17:
                raise
            pass

        file_name = self.path_temp_data + 'cclamp_' + str(amp) + '.p'

        traces = {}

        if self.force_run or (os.path.isfile(file_name) is False):
            t, v_stim, v = model.get_multiple_vm(amp, delay, dur, section_stim, loc_stim, dend_locations)

            traces['T'] = t
            traces['v_stim'] = v_stim
            traces['v_rec'] = v #dictionary key: dendritic location, value : corresponding V trace of each recording locations
            if self.save_all:
                pickle.dump(traces, gzip.GzipFile(file_name, "wb"))

        else:
            traces = pickle.load(gzip.GzipFile(file_name, "rb"))

        return traces

    def extract_somatic_spiking_features(self, traces, delay, duration):

        # soma
        trace = {}
        traces_for_efel=[]
        trace['T'] = traces['T']
        trace['V'] = traces['v_stim']
        trace['stim_start'] = [delay]
        trace['stim_end'] = [delay + duration]
        traces_for_efel.append(trace)

        # trunk locations
        '''
        for key in traces['v_rec']:
            for k in traces['v_rec'][key]:
                trace = {}
                trace['T'] = traces['T']
                trace['V'] = traces['v_rec'][key][k]
                trace['stim_start'] = [delay]
                trace['stim_end'] = [delay + duration]
                traces_for_efel.append(trace)
        '''

        efel.setDoubleSetting('interp_step', 0.025)
        efel.setDoubleSetting('DerivativeThreshold', 40.0)

        traces_results = efel.getFeatureValues(traces_for_efel, ['inv_first_ISI','AP_begin_time', 'doublet_ISI'])

        return traces_results

    def extract_amplitudes(self, traces, traces_results, actual_distances):

        #soma_AP_begin_indices = traces_results[0]['AP_begin_indices']

        soma_AP_begin_time = traces_results[0]['AP_begin_time']
        #soma_inv_first_ISI = traces_results[0]['inv_first_ISI']
        soma_first_ISI = traces_results[0]['doublet_ISI'][0]
        #print soma_AP_begin_time[0], soma_AP_begin_time[0]-1
        #print traces_results[0]['inv_first_ISI'], soma_first_ISI
        s_indices_AP1 = numpy.where(traces['T'] >= (soma_AP_begin_time[0]-1.0))
        if 10 < soma_first_ISI:
            plus = 10
        else:
            plus = soma_first_ISI-3
        e_indices_AP1 = numpy.where(traces['T'] >= (soma_AP_begin_time[0]+plus))
        start_index_AP1 = s_indices_AP1[0][0]
        end_index_AP1 = e_indices_AP1[0][0]
        #print start_index_AP1
        #print end_index_AP1

        s_indices_APlast = numpy.where(traces['T'] >= soma_AP_begin_time[-1]-1.0)
        e_indices_APlast = numpy.where(traces['T'] >= soma_AP_begin_time[-1]+10)
        start_index_APlast = s_indices_APlast[0][0]
        end_index_APlast = e_indices_APlast[0][0]

        features = collections.OrderedDict()

        for key, value in traces['v_rec'].items():
            features[key] = collections.OrderedDict()
            for k, v in traces['v_rec'][key].items():
                features[key][k] = collections.OrderedDict()

                features[key][k]['AP1_amp']= float(numpy.amax(traces['v_rec'][key][k][start_index_AP1:end_index_AP1]) - traces['v_rec'][key][k][start_index_AP1])*mV
                features[key][k]['APlast_amp']= float(numpy.amax(traces['v_rec'][key][k][start_index_APlast:end_index_APlast]) - traces['v_rec'][key][k][start_index_APlast])*mV
                features[key][k]['actual_distance'] = actual_distances[k]
        '''
        plt.figure()
        plt.plot(traces['T'],traces['v_stim'], 'r', label = 'soma')
        plt.plot(traces['T'][start_index_AP1],traces['v_stim'][start_index_AP1], 'o', label = 'soma')
        plt.plot(traces['T'][end_index_AP1],traces['v_stim'][end_index_AP1], 'o', label = 'soma')
        plt.plot(traces['T'][start_index_APlast],traces['v_stim'][start_index_APlast], 'o', label = 'soma')
        '''
        # zoom to fist AP
        plt.figure()
        plt.plot(traces['T'],traces['v_stim'], 'r', label = 'soma')
        for key, value in traces['v_rec'].items():
            for k, v in traces['v_rec'][key].items():
                #plt.plot(traces['T'],traces['v_rec'][i], label = dend_locations[i][0]+'('+str(dend_locations[i][1])+') at '+str(self.config['recording']['distances'][i])+' um')
                #plt.plot(traces['T'],traces['v_rec'][key], label = dend_locations[key][0]+'('+str(dend_locations[key][1])+') at '+str(key)+' um')
                plt.plot(traces['T'],traces['v_rec'][key][k], label = k[0]+'('+str(k[1])+') at '+str(actual_distances[k])+' um')

        plt.xlabel('time (ms)')
        plt.ylabel('membrane potential (mV)')
        plt.title('First AP')
        plt.xlim(traces['T'][start_index_AP1], traces['T'][end_index_AP1])
        lgd=plt.legend(bbox_to_anchor=(1.0, 1.0), loc = 'upper left')
        if self.save_all:
            plt.savefig(self.path_figs + 'AP1_traces'+ '.pdf', dpi=600, bbox_extra_artists=(lgd,), bbox_inches='tight')

        # zom to last AP
        plt.figure()
        plt.plot(traces['T'],traces['v_stim'], 'r', label = 'soma')
        for key, value in traces['v_rec'].items():
            for k, v in traces['v_rec'][key].items():
                #plt.plot(traces['T'],traces['v_rec'][i], label = dend_locations[i][0]+'('+str(dend_locations[i][1])+') at '+str(self.config['recording']['distances'][i])+' um')
                #plt.plot(traces['T'],traces['v_rec'][key], label = dend_locations[key][0]+'('+str(dend_locations[key][1])+') at '+str(key)+' um')
                plt.plot(traces['T'],traces['v_rec'][key][k], label = k[0]+'('+str(k[1])+') at '+str(actual_distances[k])+' um')
        plt.xlabel('time (ms)')
        plt.ylabel('membrane potential (mV)')
        plt.title('Last AP')
        plt.xlim(traces['T'][start_index_APlast], traces['T'][end_index_APlast])
        lgd=plt.legend(bbox_to_anchor=(1.0, 1.0), loc = 'upper left')
        if self.save_all:
            plt.savefig(self.path_figs + 'APlast_traces'+ '.pdf', dpi=600, bbox_extra_artists=(lgd,), bbox_inches='tight')

        return features


    def plot_traces(self, model, traces_soma_and_apical, traces_soma_and_basal, apical_locations_distances, basal_locations_distances):
        # TODO: somehow sort the traces by distance 

        if self.base_directory:
            self.path_figs = self.base_directory + 'figs/' + 'backpropagating_AP_BC/' + model.name + '/'
        else:
            self.path_figs = model.base_directory + 'figs/' + 'backpropagating_AP_BC/'


        try:
            if not os.path.exists(self.path_figs) and self.save_all:
                os.makedirs(self.path_figs)
        except OSError as e:
            if e.errno != 17:
                raise
            pass

        print("The figures are saved in the directory: ", self.path_figs)

        plt.figure(1)
        plt.plot(traces_soma_and_apical['T'],traces_soma_and_apical['v_stim'], 'r', label = 'soma')
        for key, value in traces_soma_and_apical['v_rec'].items():
            plt.plot(traces_soma_and_apical['T'],traces_soma_and_apical['v_rec'][key], label = key[0]+'('+str(key[1])+') at '+str(apical_locations_distances[key])+' um')
        plt.xlabel('time (ms)')
        plt.ylabel('membrane potential (mV)')
        lgd=plt.legend(bbox_to_anchor=(1.0, 1.0), loc = 'upper left')
        if self.save_all:
            plt.savefig(self.path_figs + 'traces_apical'+ '.pdf', dpi=600, bbox_extra_artists=(lgd,), bbox_inches='tight')

        plt.figure(2)
        plt.plot(traces_soma_and_basal['T'],traces_soma_and_basal['v_stim'], 'r', label = 'soma')
        for key, value in traces_soma_and_basal['v_rec'].items():
            plt.plot(traces_soma_and_basal['T'],traces_soma_and_basal['v_rec'][key], label = key[0]+'('+str(key[1])+') at '+str(basal_locations_distances[key])+' um')
        plt.xlabel('time (ms)')
        plt.ylabel('membrane potential (mV)')
        lgd=plt.legend(bbox_to_anchor=(1.0, 1.0), loc = 'upper left')
        if self.save_all:
            plt.savefig(self.path_figs + 'traces_basal'+ '.pdf', dpi=600, bbox_extra_artists=(lgd,), bbox_inches='tight')


    def plot_features(self, model, features, actual_distances):

        observation = self.observation

        model_AP1_amps = numpy.array([])
        model_APlast_amps = numpy.array([])
        exp_mean_AP1_amps_StrongProp = numpy.array([])
        exp_mean_AP1_amps_WeakProp = numpy.array([])
        exp_mean_APlast_amps = numpy.array([])
        exp_std_AP1_amps_StrongProp = numpy.array([])
        exp_std_AP1_amps_WeakProp = numpy.array([])
        exp_std_APlast_amps = numpy.array([])

        distances = []
        dists = numpy.array(self.config['recording']['distances'])
        location_labels = []

        for key, value in features.items():

            if 'mean_AP1_amp_strong_propagating_at_'+str(key)+'um' in list(observation.keys()) or 'mean_AP1_amp_weak_propagating_at_'+str(key)+'um' in list(observation.keys()):
                exp_mean_AP1_amps_StrongProp = numpy.append(exp_mean_AP1_amps_StrongProp, observation['mean_AP1_amp_strong_propagating_at_'+str(key)+'um'])
                exp_std_AP1_amps_StrongProp = numpy.append(exp_std_AP1_amps_StrongProp, observation['std_AP1_amp_strong_propagating_at_'+str(key)+'um'])

                exp_mean_AP1_amps_WeakProp = numpy.append(exp_mean_AP1_amps_WeakProp, observation['mean_AP1_amp_weak_propagating_at_'+str(key)+'um'])
                exp_std_AP1_amps_WeakProp = numpy.append(exp_std_AP1_amps_WeakProp, observation['std_AP1_amp_weak_propagating_at_'+str(key)+'um'])

            else:
                exp_mean_AP1_amps_WeakProp = numpy.append(exp_mean_AP1_amps_WeakProp, observation['mean_AP1_amp_at_'+str(key)+'um'])
                exp_std_AP1_amps_WeakProp = numpy.append(exp_std_AP1_amps_WeakProp, observation['std_AP1_amp_at_'+str(key)+'um'])
                exp_mean_AP1_amps_StrongProp = numpy.append(exp_mean_AP1_amps_StrongProp, observation['mean_AP1_amp_at_'+str(key)+'um'])
                exp_std_AP1_amps_StrongProp = numpy.append(exp_std_AP1_amps_StrongProp, observation['std_AP1_amp_at_'+str(key)+'um'])

            exp_mean_APlast_amps = numpy.append(exp_mean_APlast_amps, observation['mean_APlast_amp_at_'+str(key)+'um'])
            exp_std_APlast_amps = numpy.append(exp_std_APlast_amps, observation['std_APlast_amp_at_'+str(key)+'um'])

            for k, v in features[key].items() :
                distances.append(actual_distances[k])
                model_AP1_amps = numpy.append(model_AP1_amps, features[key][k]['AP1_amp'])
                model_APlast_amps = numpy.append(model_APlast_amps, features[key][k]['APlast_amp'])
                location_labels.append(k[0]+'('+str(k[1])+')')

        plt.figure()
        for i in range(len(distances)):
            plt.plot(distances[i], model_AP1_amps[i], marker ='o', linestyle='none', label = location_labels[i])
        plt.errorbar(dists, exp_mean_AP1_amps_WeakProp, yerr = exp_std_AP1_amps_WeakProp, marker='o', linestyle='none', label = 'experiment - Weak propagating')
        plt.errorbar(dists, exp_mean_AP1_amps_StrongProp, yerr = exp_std_AP1_amps_StrongProp, marker='o', linestyle='none', label = 'experiment - Strong propagating')
        plt.xlabel('Distance from soma (um)')
        plt.ylabel('AP1_amp (mV)')
        lgd = plt.legend(bbox_to_anchor=(1.0, 1.0), loc = 'upper left')
        if self.save_all:
            plt.savefig(self.path_figs + 'AP1_amps'+ '.pdf', dpi=600, bbox_extra_artists=(lgd,), bbox_inches='tight')

        plt.figure()
        for i in range(len(distances)):
            plt.plot(distances[i], model_APlast_amps[i], marker ='o', linestyle='none', label = location_labels[i])
        plt.errorbar(dists, exp_mean_APlast_amps, yerr = exp_std_APlast_amps, marker='o', linestyle='none', label = 'experiment')
        plt.xlabel('Distance from soma (um)')
        plt.ylabel('APlast_amp (mV)')
        lgd = plt.legend(bbox_to_anchor=(1.0, 1.0), loc = 'upper left')
        if self.save_all:
            plt.savefig(self.path_figs + 'APlast_amps'+ '.pdf', dpi=600, bbox_extra_artists=(lgd,), bbox_inches='tight')

    def plot_results(self, observation, prediction, errors, model_name_bAP):


        # Mean absolute feature values plot
        distances = numpy.array(self.config['recording']['distances'])

        model_mean_AP1_amps = numpy.array([])
        model_mean_APlast_amps = numpy.array([])
        model_std_AP1_amps = numpy.array([])
        model_std_APlast_amps = numpy.array([])
        exp_mean_AP1_amps_StrongProp = numpy.array([])
        exp_mean_AP1_amps_WeakProp = numpy.array([])
        exp_mean_APlast_amps = numpy.array([])
        exp_std_AP1_amps_StrongProp = numpy.array([])
        exp_std_AP1_amps_WeakProp = numpy.array([])
        exp_std_APlast_amps = numpy.array([])

        for i in range(len(distances)):
            model_mean_AP1_amps = numpy.append(model_mean_AP1_amps, prediction['model_AP1_amp_at_'+str(distances[i])+'um']['mean'])
            model_mean_APlast_amps = numpy.append(model_mean_APlast_amps, prediction['model_APlast_amp_at_'+str(distances[i])+'um']['mean'])
            model_std_AP1_amps = numpy.append(model_std_AP1_amps, prediction['model_AP1_amp_at_'+str(distances[i])+'um']['std'])
            model_std_APlast_amps = numpy.append(model_std_APlast_amps, prediction['model_APlast_amp_at_'+str(distances[i])+'um']['std'])

            if 'mean_AP1_amp_strong_propagating_at_'+str(distances[i])+'um' in list(observation.keys()) or 'mean_AP1_amp_weak_propagating_at_'+str(distances[i])+'um' in list(observation.keys()):
                exp_mean_AP1_amps_StrongProp = numpy.append(exp_mean_AP1_amps_StrongProp, observation['mean_AP1_amp_strong_propagating_at_'+str(distances[i])+'um'])
                exp_std_AP1_amps_StrongProp = numpy.append(exp_std_AP1_amps_StrongProp, observation['std_AP1_amp_strong_propagating_at_'+str(distances[i])+'um'])

                exp_mean_AP1_amps_WeakProp = numpy.append(exp_mean_AP1_amps_WeakProp, observation['mean_AP1_amp_weak_propagating_at_'+str(distances[i])+'um'])
                exp_std_AP1_amps_WeakProp = numpy.append(exp_std_AP1_amps_WeakProp, observation['std_AP1_amp_weak_propagating_at_'+str(distances[i])+'um'])

            else:
                exp_mean_AP1_amps_WeakProp = numpy.append(exp_mean_AP1_amps_WeakProp, observation['mean_AP1_amp_at_'+str(distances[i])+'um'])
                exp_std_AP1_amps_WeakProp = numpy.append(exp_std_AP1_amps_WeakProp, observation['std_AP1_amp_at_'+str(distances[i])+'um'])
                exp_mean_AP1_amps_StrongProp = numpy.append(exp_mean_AP1_amps_StrongProp, observation['mean_AP1_amp_at_'+str(distances[i])+'um'])
                exp_std_AP1_amps_StrongProp = numpy.append(exp_std_AP1_amps_StrongProp, observation['std_AP1_amp_at_'+str(distances[i])+'um'])

            exp_mean_APlast_amps = numpy.append(exp_mean_APlast_amps, observation['mean_APlast_amp_at_'+str(distances[i])+'um'])
            exp_std_APlast_amps = numpy.append(exp_std_APlast_amps, observation['std_APlast_amp_at_'+str(distances[i])+'um'])

        plt.figure()
        plt.errorbar(distances, model_mean_AP1_amps, yerr = model_std_AP1_amps, marker ='o', linestyle='none', label = model_name_bAP)
        plt.errorbar(distances, exp_mean_AP1_amps_WeakProp, yerr = exp_std_AP1_amps_WeakProp, marker='o', linestyle='none', label = 'experiment - Weak propagating')
        plt.errorbar(distances, exp_mean_AP1_amps_StrongProp, yerr = exp_std_AP1_amps_StrongProp, marker='o', linestyle='none', label = 'experiment - Strong propagating')
        plt.xlabel('Distance from soma (um)')
        plt.ylabel('AP1_amp (mV)')
        lgd=plt.legend(bbox_to_anchor=(1.0, 1.0), loc = 'upper left')
        if self.save_all:
            plt.savefig(self.path_figs + 'AP1_amp_means'+ '.pdf', dpi=600, bbox_extra_artists=(lgd,), bbox_inches='tight')

        plt.figure()
        plt.errorbar(distances, model_mean_APlast_amps, yerr = model_std_APlast_amps, marker ='o', linestyle='none', label = model_name_bAP)
        plt.errorbar(distances, exp_mean_APlast_amps, yerr = exp_std_APlast_amps, marker='o', linestyle='none', label = 'experiment')
        plt.xlabel('Distance from soma (um)')
        plt.ylabel('APlast_amp (mV)')
        lgd=plt.legend(bbox_to_anchor=(1.0, 1.0), loc = 'upper left')
        if self.save_all:
            plt.savefig(self.path_figs + 'APlast_amp_means'+ '.pdf', dpi=600, bbox_extra_artists=(lgd,), bbox_inches='tight')


        # Plot of errors

        keys = []
        values = []

        #fig, ax = plt.subplots()
        plt.figure()
        for key, value in errors.items():
            keys.append(key)
            values.append(value)
        y=list(range(len(keys)))
        y.reverse()
        #ax.set_yticks(y)
        #print keys
        #print values
        plt.plot(values, y, 'o')
        plt.yticks(y, keys)
        if self.save_all:
            plt.savefig(self.path_figs + 'bAP_errors'+ '.pdf', bbox_inches='tight')
    '''
    def validate_observation(self, observation):

        for key, value in observation.items():
            try:
                assert type(observation[key]) is Quantity
            except Exception as e:
                raise ObservationError(("Observation must be of the form "
                                        "{'mean':float*mV,'std':float*mV}"))
    '''

    def generate_prediction(self, model, verbose=False):
        """Implementation of sciunit.Test.generate_prediction."""

        efel.reset()

        if self.base_directory:
            self.path_results = self.base_directory + 'results/' + 'backpropagating_AP_BC/' + model.name + '/'
        else:
            self.path_results = model.base_directory + 'results/' + 'backpropagating_AP_BC/'

        try:
            if not os.path.exists(self.path_results):
                os.makedirs(self.path_results)
        except OSError as e:
            if e.errno != 17:
                raise
            pass


        global model_name_bAP
        model_name_bAP = model.name

        distances_apical = self.config['apical_recording']['distances']
        tolerance_apical = self.config['apical_recording']['tolerance']
        dist_range_apical = [min(distances_apical) - tolerance_apical, max(distances_apical) + tolerance_apical]
        distances_basal = self.config['basal_recording']['distances']
        tolerance_basal = self.config['basal_recording']['tolerance']
        dist_range_basal = [min(distances_basal) - tolerance_basal, max(distances_basal) + tolerance_basal]

        apical_locations, apical_locations_distances = model.get_random_locations_multiproc('trunk', self.num_of_apical_dend_locations, self.random_seed, dist_range_apical) # number of random locations , seed
        # apical_locations_distances = sorted(apical_locations_distances, key=apical_locations_distances.get)
        #print dend_locations, actual_distances
        print('Apical dendritic locations to be tested (with their actual distances):', apical_locations_distances)

        basal_locations, basal_locations_distances = model.get_random_locations_multiproc('basal', self.num_of_basal_dend_locations, self.random_seed, dist_range_basal) # number of random locations , seed
        # basal_locations_distances = sorted(basal_locations_distances, key=basal_locations_distances.get)
        #print dend_locations, actual_distances
        print('Basal dendritic locations to be tested (with their actual distances):', basal_locations_distances)

        dend_locations = apical_locations + basal_locations   # so the simulation is run only once, and record from alll the locations at the same time 

        traces={}
        delay = self.config['stimulus']['delay']
        duration = self.config['stimulus']['duration']
        amplitude = self.config['stimulus']['amplitude']

        prediction = collections.OrderedDict()

        plt.close('all') #needed to avoid overlapping of saved images when the test is run on multiple models


        pool = multiprocessing.Pool(1, maxtasksperchild = 1)
        traces = pool.apply(self.cclamp, args = (model, amplitude, delay, duration, "soma", 0.5, dend_locations))
        print(traces)

        traces_soma_and_basal = collections.OrderedDict() 
        traces_soma_and_basal['T'] = traces['T'] 
        traces_soma_and_basal['v_stim'] = traces['v_stim']  
        traces_soma_and_basal['v_rec'] = collections.OrderedDict()

        traces_soma_and_apical = collections.OrderedDict() 
        traces_soma_and_apical['T'] = traces['T'] 
        traces_soma_and_apical['v_stim'] = traces['v_stim']  
        traces_soma_and_apical['v_rec'] =collections.OrderedDict()

        for key, value in traces['v_rec'].items():
            if list(key) in apical_locations:
               traces_soma_and_apical['v_rec'].update({key:value}) 
            if list(key) in basal_locations:
               traces_soma_and_basal['v_rec'].update({key:value})
        print(traces_soma_and_apical)
        print(traces_soma_and_basal)       

        filepath = self.path_results + self.test_log_filename
        self.logFile = open(filepath, 'w') # if it is opened before multiprocessing, the multiporeccing won't work under python3

        self.logFile.write('Apical dendritic locations to be tested (with their actual distances):\n'+ str(apical_locations_distances)+'\n')
        self.logFile.write('Basal dendritic locations to be tested (with their actual distances):\n'+ str(basal_locations_distances)+'\n')
        self.logFile.write("---------------------------------------------------------------------------------------------------\n")

        # self.logFile.write(message_to_logFile)



        self.plot_traces(model, traces_soma_and_apical, traces_soma_and_basal, apical_locations_distances, basal_locations_distances)

        """itt tartok"""
        traces_results = self.extract_somatic_spiking_features(traces, delay, duration)


        features = self.extract_amplitudes(traces, traces_results, actual_distances)

        features_json = collections.OrderedDict()
        for key in features:
            features_json[key] = collections.OrderedDict()
            for ke in features[key]:
                features_json[key][str(ke)] = collections.OrderedDict()
                for k, value in features[key][ke].items():
                    features_json[key][str(ke)][k] = str(value)


        # generating prediction
        for key in features:
            AP1_amps = numpy.array([])
            APlast_amps = numpy.array([])

            for k in features[key]:
                AP1_amps = numpy.append(AP1_amps, features[key][k]['AP1_amp'] )
            prediction['model_AP1_amp_at_'+str(key)+'um'] = {}
            prediction['model_AP1_amp_at_'+str(key)+'um']['mean'] = float(numpy.mean(AP1_amps))*mV
            prediction['model_AP1_amp_at_'+str(key)+'um']['std'] = float(numpy.std(AP1_amps))*mV

        for key in features:
            AP1_amps = numpy.array([])
            APlast_amps = numpy.array([])
            for k in features[key]:
                APlast_amps = numpy.append(APlast_amps, features[key][k]['APlast_amp'] )
            prediction['model_APlast_amp_at_'+str(key)+'um'] = {}
            prediction['model_APlast_amp_at_'+str(key)+'um']['mean'] = float(numpy.mean(APlast_amps))*mV
            prediction['model_APlast_amp_at_'+str(key)+'um']['std'] = float(numpy.std(APlast_amps))*mV

        prediction_json = collections.OrderedDict()
        for key in prediction:
            prediction_json[key] = collections.OrderedDict()
            for k, value in prediction[key].items():
                prediction_json[key][k]=str(value)


        file_name_json = self.path_results + 'bAP_model_features_means.json'
        json.dump(prediction_json, open(file_name_json, "w"), indent=4)
        file_name_features_json = self.path_results + 'bAP_model_features.json'
        json.dump(features_json, open(file_name_features_json, "w"), indent=4)

        if self.save_all:
            file_name_pickle = self.path_results + 'bAP_model_features.p'

            pickle.dump(features, gzip.GzipFile(file_name_pickle, "wb"))

            file_name_pickle = self.path_results + 'bAP_model_features_means.p'

            pickle.dump(prediction, gzip.GzipFile(file_name_pickle, "wb"))

        self.plot_features(model, features, actual_distances)

        efel.reset()

        return prediction

    def compute_score(self, observation, prediction, verbose=False):
        """Implementation of sciunit.Test.score_prediction."""

        distances = numpy.array(self.config['recording']['distances'])
        #score_sum_StrongProp, score_sum_WeakProp  = scores.ZScore_backpropagatingAP.compute(observation,prediction, [50, 150, 250])
        score_avg, errors= scores.ZScore_backpropagatingAP.compute(observation,prediction, distances)

        scores_dict = {}
        scores_dict['Z_score_avg_strong_propagating'] = score_avg[0]
        scores_dict['Z_score_avg_weak_propagating'] = score_avg[1]

        file_name=self.path_results+'bAP_errors.json'

        json.dump(errors, open(file_name, "w"), indent=4)

        file_name_s=self.path_results+'bAP_scores.json'

        json.dump(scores_dict, open(file_name_s, "w"), indent=4)

        self.plot_results(observation, prediction, errors, model_name_bAP)

        if self.show_plot:
            plt.show()

        if scores.ZScore_backpropagatingAP.strong:#score_avg[0] < score_avg[1]:
            best_score = score_avg[0]
            print('This is a rather STRONG propagating model')


            self.logFile.write('This is a rather STRONG propagating model\n')
            self.logFile.write("---------------------------------------------------------------------------------------------------\n")

            score_json= {'Z_score_avg_STRONG_propagating' : best_score}
        elif scores.ZScore_backpropagatingAP.strong is False:#score_avg[1] < score_avg[0]:
            best_score = score_avg[1]
            print('This is a rather WEAK propagating model')

            self.logFile.write('This is a rather WEAK propagating model\n')
            self.logFile.write("---------------------------------------------------------------------------------------------------\n")

            score_json= {'Z_score_avg_Weak_propagating' : best_score}
        elif scores.ZScore_backpropagatingAP.strong is None:#score_avg[1] == score_avg[0]:
            best_score = score_avg[0]
            score_json= {'Z_score_avg' : best_score}


        file_name_score = self.path_results + 'bAP_final_score.json'
        json.dump(score_json, open(file_name_score, "w"), indent=4)


        score=scores.ZScore_backpropagatingAP(best_score)

        self.logFile.write(str(score)+'\n')
        self.logFile.write("---------------------------------------------------------------------------------------------------\n")


        self.logFile.close()

        self.logFile = self.path_results + self.test_log_filename

        return score

    def bind_score(self, score, model, observation, prediction):

        score.related_data["figures"] = [self.path_figs + 'AP1_amp_means.pdf', self.path_figs + 'AP1_amps.pdf', self.path_figs + 'AP1_traces.pdf',
                                        self.path_figs + 'APlast_amp_means.pdf', self.path_figs + 'APlast_amps.pdf', self.path_figs + 'APlast_traces.pdf', self.path_figs + 'Spikecounts_bAP.pdf',
                                        self.path_figs + 'bAP_errors.pdf', self.path_figs + 'traces.pdf', self.path_results + 'bAP_errors.json',
                                        self.path_results + 'bAP_model_features.json', self.path_results + 'bAP_model_features_means.json',
                                        self.path_results + 'bAP_scores.json', self.path_results + 'bAP_final_score.json', self.path_results + self.test_log_filename]
        score.related_data["results"] = [self.path_results + 'bAP_errors.json', self.path_results + 'bAP_model_features.json', self.path_results + 'bAP_model_features_means.json', self.path_results + 'bAP_scores.json', self.path_results + 'bAP_model_features.p', self.path_results + 'bAP_model_features_means.p', self.path_results + 'bAP_final_score.json']
        return score
