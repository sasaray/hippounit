from __future__ import print_function
from __future__ import division
from future import standard_library
standard_library.install_aliases()
#from builtins import str
from builtins import range
from quantities.quantity import Quantity
from quantities import mV, nA, V, ms
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


class BackpropagatingAPTest_CA3_PC(Test):
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
                                        cap.ProvidesRandomDendriticLocations, cap.ReceivesSquareCurrent_ProvidesResponse, 
                                         cap.ReceivesCurrentPulses_ProvidesResponse_MultipleLocations,)

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

        description = "Tests efficacy  and shape of back-propagating action potentials on the basal and apical dendrites of hippocampal CA3 pyramidal cells."

    score_type = scores.ZScore_backpropagatingAP_BasketCell

    def format_data(self, observation):
        for loc in list(observation.keys()):
            if loc == 'soma':
                for key, val in list(observation[loc].items()):
                    if key == 'long input':
                         for ke, va in list(observation[loc][key].items()): 
                             try:
                                 assert type(observation[loc][key][ke]['mean']) is Quantity
                                 assert type(observation[loc][key][ke]['std']) is Quantity
                             except Exception as e:
                                 quantity_parts = va['mean'].split(" ")
                                 number = float(quantity_parts[0])
                                 units = " ".join(quantity_parts[1:])
                                 observation[loc][key][ke]['mean'] = Quantity(number, units)

                                 quantity_parts = va['std'].split(" ")
                                 number = float(quantity_parts[0])
                                 units = " ".join(quantity_parts[1:])
                                 observation[loc][key][ke]['std'] = Quantity(number, units)
                    else: 
                        for ke, va in list(observation[loc][key].items()): 
                            for k, v in list(observation[loc][key][ke].items()): 
                                try:
                                    assert type(observation[loc][key][ke][k]['mean']) is Quantity
                                    assert type(observation[loc][key][ke][k]['std']) is Quantity
                                except Exception as e:
                                    quantity_parts = v['mean'].split(" ")
                                    number = float(quantity_parts[0])
                                    units = " ".join(quantity_parts[1:])
                                    observation[loc][key][ke][k]['mean'] = Quantity(number, units)

                                    quantity_parts = v['std'].split(" ")
                                    number = float(quantity_parts[0])
                                    units = " ".join(quantity_parts[1:])
                                    observation[loc][key][ke][k]['std'] = Quantity(number, units)

            else:
                for key, val in list(observation[loc].items()):
                    if key == 'long input':
                         for dist in list(observation[loc][key].keys()):
                             for ke, va in list(observation[loc][key][dist].items()): 
                                 try:
                                     assert type(observation[loc][key][dist][ke]['mean']) is Quantity
                                     assert type(observation[loc][key][dist][ke]['std']) is Quantity
                                 except Exception as e:
                                     quantity_parts = va['mean'].split(" ")
                                     number = float(quantity_parts[0])
                                     units = " ".join(quantity_parts[1:])
                                     observation[loc][key][dist][ke]['mean'] = Quantity(number, units)

                                     quantity_parts = va['std'].split(" ")
                                     number = float(quantity_parts[0])
                                     units = " ".join(quantity_parts[1:])
                                     observation[loc][key][dist][ke]['std'] = Quantity(number, units)
                    else: 
                        for dist in list(observation[loc][key].keys()):
                            for ke, va in list(observation[loc][key][dist].items()): 
                                for k, v in list(observation[loc][key][dist][ke].items()): 
                                    try:
                                        assert type(observation[loc][key][dist][ke][k]['mean']) is Quantity
                                        assert type(observation[loc][key][dist][ke][k]['std']) is Quantity
                                    except Exception as e:
                                        quantity_parts = v['mean'].split(" ")
                                        number = float(quantity_parts[0])
                                        units = " ".join(quantity_parts[1:])
                                        observation[loc][key][dist][ke][k]['mean'] = Quantity(number, units)

                                        quantity_parts = v['std'].split(" ")
                                        number = float(quantity_parts[0])
                                        units = " ".join(quantity_parts[1:])
                                        observation[loc][key][dist][ke][k]['std'] = Quantity(number, units)
        # print(observation)
        return observation


    def long_rheobase_input(self, model, amp, delay, dur, section_stim, loc_stim, dend_locations):

        if self.base_directory:
            self.path_temp_data = self.base_directory + 'temp_data/' + 'backpropagating_AP_CA3_PC/' + model.name + '/'
        else:
            self.path_temp_data = model.base_directory + 'temp_data/' + 'backpropagating_AP_CA3_PC/'


        try:
            if not os.path.exists(self.path_temp_data) and self.save_all:
                os.makedirs(self.path_temp_data)
        except OSError as e:
            if e.errno != 17:
                raise
            pass

        file_name = self.path_temp_data + 'long_rheobase_input_' + str(amp) + '.p'

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

    def current_pulses(self, model, amp, delay, dur_of_pulse, dur_of_stim, num_of_pulses, section_stim, loc_stim, dend_locations, frequency):

        if self.base_directory:
            self.path_temp_data = self.base_directory + 'temp_data/' + 'backpropagating_AP_CA3_PC/' + model.name + '/'
        else:
            self.path_temp_data = model.base_directory + 'temp_data/' + 'backpropagating_AP_CA3_PC/'


        try:
            if not os.path.exists(self.path_temp_data) and self.save_all:
                os.makedirs(self.path_temp_data)
        except OSError as e:
            if e.errno != 17:
                raise
            pass

        file_name = self.path_temp_data + 'pulses_' + str(frequency) + 'Hz.p'

        traces = {}

        if self.force_run or (os.path.isfile(file_name) is False):
            t, v_stim, v = model.inject_current_pulses_get_multiple_vm(amp, delay, dur_of_pulse, dur_of_stim, num_of_pulses, frequency, section_stim, loc_stim, dend_locations)
            traces['T'] = t
            traces['v_stim'] = v_stim
            traces['v_rec'] = v #dictionary key: dendritic location, value : corresponding V trace of each recording locations
            if self.save_all:
                pickle.dump(traces, gzip.GzipFile(file_name, "wb"))

        else:
            traces = pickle.load(gzip.GzipFile(file_name, "rb"))

        return traces

    def extract_features_by_eFEL(self, traces_soma_and_apical, traces_soma_and_basal, delay, dur_of_stim, delay_long, duration_long, frequencies):



        traces_results_soma = {'train of brief inputs' : {}, 'long input' : None} 
        traces_results_apical = {'train of brief inputs' : {}, 'long input' : {}}
        traces_results_basal = {'train of brief inputs' : {}, 'long input' : {}}


        ''' brief pulses stim'''
        for freq in frequencies: 

            traces_results_soma['train of brief inputs'][freq] = {} 
            traces_results_apical['train of brief inputs'][freq] = {} 
            traces_results_basal['train of brief inputs'][freq] = {} 

            # soma
            trace_soma = {}
            traces_soma=[]
            trace_soma['T'] = traces_soma_and_apical['train of brief inputs'][freq]['T']
            trace_soma['V'] = traces_soma_and_apical['train of brief inputs'][freq]['v_stim']
            trace_soma['stim_start'] = [delay]
            trace_soma['stim_end'] = [delay + dur_of_stim]
            traces_soma.append(trace_soma)
            traces_results_soma['train of brief inputs'][freq] = efel.getFeatureValues(traces_soma, ['AP_amplitude','AP_rise_rate', 'AP_duration_half_width', 'inv_first_ISI','AP_begin_time', 'doublet_ISI'])

            #apical
            for key, value in traces_soma_and_apical['train of brief inputs'][freq]['v_rec'].items():
                trace = {}
                traces_for_efel=[]
                trace['T'] = traces_soma_and_apical['train of brief inputs'][freq]['T']
                trace['V'] = value
                trace['stim_start'] = [delay]
                trace['stim_end'] = [delay + dur_of_stim]
                traces_for_efel.append(trace)
                traces_results = efel.getFeatureValues(traces_for_efel, ['AP_amplitude', 'AP_duration_half_width'])
                traces_results_apical['train of brief inputs'][freq].update({key : traces_results}) 

            #basal
            for key, value in traces_soma_and_basal['train of brief inputs'][freq]['v_rec'].items():
                trace = {}
                traces_for_efel=[]
                trace['T'] = traces_soma_and_basal['train of brief inputs'][freq]['T']
                trace['V'] = value
                trace['stim_start'] = [delay]
                trace['stim_end'] = [delay + dur_of_stim]
                traces_for_efel.append(trace)


                traces_results = efel.getFeatureValues(traces_for_efel, ['AP_amplitude', 'AP_duration_half_width'])
                traces_results_basal['train of brief inputs'][freq].update({key : traces_results})


        ''' long input '''
        # soma
        trace_soma = {}
        traces_soma=[]
        trace_soma['T'] = traces_soma_and_apical['long input']['T']
        trace_soma['V'] = traces_soma_and_apical['long input']['v_stim']
        trace_soma['stim_start'] = [delay_long]
        trace_soma['stim_end'] = [delay_long + duration_long]
        traces_soma.append(trace_soma)


        traces_results_soma['long input'] = efel.getFeatureValues(traces_soma, ['AP_amplitude', 'AP_duration_half_width', 'inv_first_ISI','AP_begin_time', 'doublet_ISI'])
   
        #apical
        for key, value in traces_soma_and_apical['long input']['v_rec'].items():
            trace = {}
            traces_for_efel=[]
            trace['T'] = traces_soma_and_apical['long input']['T']
            trace['V'] = value
            trace['stim_start'] = [delay_long]
            trace['stim_end'] = [delay_long + duration_long]
            traces_for_efel.append(trace)


            traces_results = efel.getFeatureValues(traces_for_efel, ['AP_amplitude', 'AP_duration_half_width'])
            traces_results_apical['long input'][key] = traces_results

        #basal 
        for key, value in traces_soma_and_basal['long input']['v_rec'].items():
            trace = {}
            traces_for_efel=[]
            trace['T'] = traces_soma_and_basal['long input']['T']
            trace['V'] = value
            trace['stim_start'] = [delay_long]
            trace['stim_end'] = [delay_long + duration_long]
            traces_for_efel.append(trace)


            traces_results = efel.getFeatureValues(traces_for_efel, ['AP_amplitude', 'AP_duration_half_width'])
            traces_results_basal['long input'][key] = traces_results 


        return traces_results_soma, traces_results_apical, traces_results_basal


    def get_time_indices_befor_and_after_somatic_AP(self, efel_features_somatic, traces_soma_and_apical, frequencies):

        start_index_AP1 = {} 
        end_index_AP1 = {}
        start_index_AP5 = {}
        end_index_AP5 = {}

        for freq in frequencies:
            soma_AP_begin_time = efel_features_somatic['train of brief inputs'][freq][0]['AP_begin_time']
            soma_first_ISI = efel_features_somatic['train of brief inputs'][freq][0]['doublet_ISI'][0]
            s_indices_AP1 = numpy.where(traces_soma_and_apical['train of brief inputs'][freq]['T'] >= (soma_AP_begin_time[0]-1.0))
            if 10 < soma_first_ISI:
                plus = 10
            else:
                plus = soma_first_ISI-3
            e_indices_AP1 = numpy.where(traces_soma_and_apical['train of brief inputs'][freq]['T'] >= (soma_AP_begin_time[0]+plus))
            start_index_AP1[freq] = s_indices_AP1[0][0]
            end_index_AP1[freq] = e_indices_AP1[0][0]
            #print start_index_AP1
            #print end_index_AP1

            s_indices_AP5 = numpy.where(traces_soma_and_apical['train of brief inputs'][freq]['T'] >= soma_AP_begin_time[4]-1.0)
            e_indices_AP5 = numpy.where(traces_soma_and_apical['train of brief inputs'][freq]['T'] >= soma_AP_begin_time[4]+10)
            start_index_AP5[freq] = s_indices_AP5[0][0]
            end_index_AP5[freq] = e_indices_AP5[0][0]
        print (start_index_AP1, end_index_AP1, start_index_AP5, end_index_AP5)
        return [start_index_AP1, end_index_AP1, start_index_AP5, end_index_AP5]  

    def plot_AP1_AP5(self, time_indices_befor_and_after_somatic_AP, apical_locations_distances, basal_locations_distances, traces_soma_and_apical, traces_soma_and_basal, frequencies):

        start_index_AP1, end_index_AP1, start_index_AP5, end_index_AP5 = time_indices_befor_and_after_somatic_AP

        # zoom to first and fifth AP apical
        fig1, axs1 = plt.subplots(len(frequencies),2, sharey = True)
        plt.subplots_adjust(wspace = 0.4, hspace = 0.4 )
        for i, freq in enumerate(frequencies):
            axs1[i, 0].plot(traces_soma_and_apical['train of brief inputs'][freq]['T'],traces_soma_and_apical['train of brief inputs'][freq]['v_stim'], 'r')
            axs1[i, 1].plot(traces_soma_and_apical['train of brief inputs'][freq]['T'],traces_soma_and_apical['train of brief inputs'][freq]['v_stim'], 'r', label = 'soma')
            axs1[i, 0].text(0.95,0.9, str(freq) + ' Hz', horizontalalignment='right', verticalalignment='top', transform=axs1[i,0].transAxes)
            axs1[i, 1].text(0.95,0.9, str(freq) + ' Hz', horizontalalignment='right', verticalalignment='top', transform=axs1[i,1].transAxes)
            for key, value in traces_soma_and_apical['train of brief inputs'][freq]['v_rec'].items():
                axs1[i, 0].plot(traces_soma_and_apical['train of brief inputs'][freq]['T'],traces_soma_and_apical['train of brief inputs'][freq]['v_rec'][key])
                axs1[i, 1].plot(traces_soma_and_apical['train of brief inputs'][freq]['T'],traces_soma_and_apical['train of brief inputs'][freq]['v_rec'][key], label = key[0]+'('+str(key[1])+') at '+str(apical_locations_distances[key])+' um')

            axs1[i, 0].set_xlim(traces_soma_and_apical['train of brief inputs'][freq]['T'][start_index_AP1[freq]], traces_soma_and_apical['train of brief inputs'][freq]['T'][end_index_AP1[freq]])
            axs1[i, 1].set_xlim(traces_soma_and_apical['train of brief inputs'][freq]['T'][start_index_AP5[freq]], traces_soma_and_apical['train of brief inputs'][freq]['T'][end_index_AP5[freq]])
            axs1[i, 0].set_ylabel('membrane\n  potential (mV)')
        axs1[0, 0].set_title('First AP')
        axs1[-1, 0].set_xlabel('time (ms)')
        axs1[-1, 1].set_xlabel('time (ms)')
        axs1[0, 1].set_title('Fifth AP')
            
        lgd=axs1[0, 1].legend(bbox_to_anchor=(1.0, 1.0), loc = 'upper left')
        fig1.suptitle('Apical dendrites')
        if self.save_all:
            plt.savefig(self.path_figs + 'First_and_fifth_APs_apical'+ '.pdf', dpi=600, bbox_extra_artists=(lgd,), bbox_inches='tight')

        # zoom to first and fifth AP basal
        fig2, axs2 = plt.subplots(len(frequencies),2, sharey=True)
        plt.subplots_adjust(wspace = 0.4, hspace = 0.6)
        for i, freq in enumerate(frequencies):
            axs2[i, 0].plot(traces_soma_and_basal['train of brief inputs'][freq]['T'],traces_soma_and_basal['train of brief inputs'][freq]['v_stim'], 'r')
            axs2[i, 1].plot(traces_soma_and_basal['train of brief inputs'][freq]['T'],traces_soma_and_basal['train of brief inputs'][freq]['v_stim'], 'r', label = 'soma')
            axs2[i, 0].text(0.95,0.9, str(freq) + ' Hz', horizontalalignment='right', verticalalignment='top', transform=axs1[i,0].transAxes)
            axs2[i, 1].text(0.95,0.9, str(freq) + ' Hz', horizontalalignment='right', verticalalignment='top', transform=axs1[i,1].transAxes)
            for key, value in traces_soma_and_basal['train of brief inputs'][freq]['v_rec'].items():
                axs2[i, 0].plot(traces_soma_and_basal['train of brief inputs'][freq]['T'],traces_soma_and_basal['train of brief inputs'][freq]['v_rec'][key])
                axs2[i, 1].plot(traces_soma_and_basal['train of brief inputs'][freq]['T'],traces_soma_and_basal['train of brief inputs'][freq]['v_rec'][key], label = key[0]+'('+str(key[1])+') at '+str(basal_locations_distances[key])+' um')
            axs2[i, 0].set_xlim(traces_soma_and_basal['train of brief inputs'][freq]['T'][start_index_AP1[freq]], traces_soma_and_basal['train of brief inputs'][freq]['T'][end_index_AP1[freq]])
            axs2[i, 1].set_xlim(traces_soma_and_basal['train of brief inputs'][freq]['T'][start_index_AP5[freq]], traces_soma_and_basal['train of brief inputs'][freq]['T'][end_index_AP5[freq]])
            axs2[i, 0].set_ylabel('membrane\n potential (mV)')
        axs2[0, 0].set_title('First AP')
        axs2[-1, 0].set_xlabel('time (ms)')
        axs2[-1, 1].set_xlabel('time (ms)')
        axs2[0, 1].set_title('Fifth AP')

        lgd=axs2[0, 1].legend(bbox_to_anchor=(1.0, 1.0), loc = 'upper left')
        fig2.suptitle('Basal dendrites')
        if self.save_all:
            plt.savefig(self.path_figs + 'First_and_fifth_APs_basal'+ '.pdf', dpi=600, bbox_extra_artists=(lgd,), bbox_inches='tight')


    def extract_prediction_features(self, efel_features_somatic, efel_features_apical, efel_features_basal, apical_locations_distances, basal_locations_distances, distances_apical, tolerance_apical, distances_basal, tolerance_basal, frequencies):

        features = {'soma' : {'long input' : {'AP amplitude' : efel_features_somatic['long input'][0]['AP_amplitude'][0]*mV,
                                              'AP half-duration' : efel_features_somatic['long input'][0]['AP_duration_half_width'][0]*ms},
                              'train of brief inputs' : {}},
                    'apical' : {'long input' : {}, 
                                'train of brief inputs' :{}},
                    'basal' : {'long input' : {}, 
                               'train of brief inputs' :{}}  
                    }  

        features_to_json = {'soma' : {'long input' : {'AP amplitude' : str(efel_features_somatic['long input'][0]['AP_amplitude'][0]*mV),
                                                      'AP half-duration' : str(efel_features_somatic['long input'][0]['AP_duration_half_width'][0]*ms)},
                                      'train of brief inputs' :{}},
                    'apical' : {'long input' : {},
                                'train of brief inputs' :{}},
                    'basal' : {'long input' : {}, 
                               'train of brief inputs' :{}}  
                    } 

        for key, value in efel_features_apical['long input'].items():
            features['apical']['long input'].update({key : {}}) 
            features['apical']['long input'][key].update({'AP amplitude' : value[0]['AP_amplitude'][0]*mV}) 
            features['apical']['long input'][key].update({'AP half-duration' : value[0]['AP_duration_half_width'][0]*mV})
            features['apical']['long input'][key].update({'distance' :apical_locations_distances[key]})

            features_to_json['apical']['long input'].update({str(key) : {}}) 
            features_to_json['apical']['long input'][str(key)].update({'AP amplitude' : str(value[0]['AP_amplitude'][0]*mV)}) 
            features_to_json['apical']['long input'][str(key)].update({'AP half-duration' : str(value[0]['AP_duration_half_width'][0]*mV)})
            features_to_json['apical']['long input'][str(key)].update({'distance' :apical_locations_distances[key]})

        for key, value in efel_features_basal['long input'].items():
            features['basal']['long input'].update({key : {}}) 
            features['basal']['long input'][key].update({'AP amplitude' : value[0]['AP_amplitude'][0]*mV}) 
            features['basal']['long input'][key].update({'AP half-duration' : value[0]['AP_duration_half_width'][0]*mV})
            features['basal']['long input'][key].update({'distance' : basal_locations_distances[key]})

            features_to_json['basal']['long input'].update({str(key) : {}}) 
            features_to_json['basal']['long input'][str(key)].update({'AP amplitude' : str(value[0]['AP_amplitude'][0]*mV)}) 
            features_to_json['basal']['long input'][str(key)].update({'AP half-duration' : str(value[0]['AP_duration_half_width'][0]*mV)})
            features_to_json['basal']['long input'][str(key)].update({'distance' : basal_locations_distances[key]})

        for freq in frequencies:
            features['soma']['train of brief inputs'][str(freq) + ' Hz'] = {} 
            features['apical']['train of brief inputs'][str(freq) + ' Hz'] = {} 
            features['basal']['train of brief inputs'][str(freq) + ' Hz'] = {} 

            features_to_json['soma']['train of brief inputs'][str(freq) + ' Hz'] = {} 
            features_to_json['apical']['train of brief inputs'][str(freq) + ' Hz'] = {} 
            features_to_json['basal']['train of brief inputs'][str(freq) + ' Hz'] = {} 

            features['soma']['train of brief inputs'][str(freq) + ' Hz'].update({'AP amplitude-AP5/AP1' : efel_features_somatic['train of brief inputs'][freq][0]['AP_amplitude'][4] / efel_features_somatic['train of brief inputs'][freq][0]['AP_amplitude'][0]})
            features['soma']['train of brief inputs'][str(freq) + ' Hz'].update({'AP half-duration-AP5/AP1' : efel_features_somatic['train of brief inputs'][freq][0]['AP_duration_half_width'][4] / efel_features_somatic['train of brief inputs'][freq][0]['AP_duration_half_width'][0]}) 
            
            features_to_json['soma']['train of brief inputs'][str(freq) + ' Hz'].update({'AP amplitude-AP5/AP1' : str(efel_features_somatic['train of brief inputs'][freq][0]['AP_amplitude'][4] / efel_features_somatic['train of brief inputs'][freq][0]['AP_amplitude'][0])})
            features_to_json['soma']['train of brief inputs'][str(freq) + ' Hz'].update({'AP half-duration-AP5/AP1' : str(efel_features_somatic['train of brief inputs'][freq][0]['AP_duration_half_width'][4] / efel_features_somatic['train of brief inputs'][freq][0]['AP_duration_half_width'][0])}) 

            for key, value in efel_features_apical['train of brief inputs'][freq].items():
                features['apical']['train of brief inputs'][str(freq) + ' Hz'].update({key :{ 'distance' : apical_locations_distances[key], 
                                              'AP amplitude-AP5/AP1' : value[0]['AP_amplitude'][4] / value[0]['AP_amplitude'][0],
                                              'AP half-duration-AP5/AP1' : value[0]['AP_duration_half_width'][4] / value[0]['AP_duration_half_width'][0],
                                             } 
                                        })

                features_to_json['apical']['train of brief inputs'][str(freq) + ' Hz'].update({str(key) :{ 'distance' : apical_locations_distances[key], 
                                              'AP amplitude-AP5/AP1' : str(value[0]['AP_amplitude'][4] / value[0]['AP_amplitude'][0]),
                                              'AP half-duration-AP5/AP1' : str(value[0]['AP_duration_half_width'][4] / value[0]['AP_duration_half_width'][0]),
                                             } 
                                        })
 
            for key, value in efel_features_basal['train of brief inputs'][freq].items():
                features['basal']['train of brief inputs'][str(freq) + ' Hz'].update({key:{ 'distance' : basal_locations_distances[key], 
                                              'AP amplitude-AP5/AP1' : value[0]['AP_amplitude'][4] / value[0]['AP_amplitude'][0],
                                              'AP half-duration-AP5/AP1' : value[0]['AP_duration_half_width'][4] / value[0]['AP_duration_half_width'][0],
                                             } 
                                        })

                features_to_json['basal']['train of brief inputs'][str(freq) + ' Hz'].update({str(key) :{ 'distance' : basal_locations_distances[key], 
                                              'AP amplitude-AP5/AP1' : str(value[0]['AP_amplitude'][4] / value[0]['AP_amplitude'][0]),
                                              'AP half-duration-AP5/AP1' : str(value[0]['AP_duration_half_width'][4] / value[0]['AP_duration_half_width'][0]),
                                             } 
                                        })
        prediction = {'soma' : features['soma'],
                      'apical' : {'long input' : {}, 
                                  'train of brief inputs' :{}},
                      'basal' : {'long input' : {}, 
                                 'train of brief inputs' :{}}  
                    }  

        prediction_to_json = {'soma': features_to_json['soma'], 
                              'apical' : {'long input' : {}, 
                                          'train of brief inputs' :{}},
                               'basal' : {'long input' : {}, 
                                          'train of brief inputs' :{}}  
                              } 

        for dist in distances_apical:
            AP_amps = numpy.array([])
            AP_half_durs = numpy.array([])


            prediction['apical']['long input'].update({dist : collections.OrderedDict()})
            prediction_to_json['apical']['long input'].update({dist : collections.OrderedDict()})

            for key, value in features['apical']['long input'].items():
 
                if value['distance'] >= dist - tolerance_apical and  value['distance'] < dist + tolerance_apical:
                    

                    AP_amps = numpy.append(AP_amps, value['AP amplitude'])
                    AP_half_durs = numpy.append(AP_half_durs, value['AP half-duration']) 

            prediction['apical']['long input'][dist].update({  
                                              'AP amplitude' : {'mean': numpy.mean(AP_amps)*mV, 'std' : numpy.std(AP_amps)*mV}, 
                                              'AP half-duration' : {'mean': numpy.mean(AP_half_durs)*ms , 'std' : numpy.std(AP_half_durs)*ms}
                                             })
            prediction_to_json['apical']['long input'][dist].update({  
                                              'AP amplitude' : {'mean': str(numpy.mean(AP_amps)*mV), 'std' : str(numpy.std(AP_amps)*mV)}, 
                                              'AP half-duration' : {'mean': str(numpy.mean(AP_half_durs)*ms) , 'std' : str(numpy.std(AP_half_durs)*ms)}
                                             })

        for dist in distances_basal:
            AP_amps = numpy.array([])
            AP_half_durs = numpy.array([])


            prediction['basal']['long input'].update({dist : collections.OrderedDict()})
            prediction_to_json['basal']['long input'].update({dist : collections.OrderedDict()})

            for key, value in features['basal']['long input'].items():
 
                if value['distance'] >= dist - tolerance_basal and  value['distance'] < dist + tolerance_basal:
      

                    AP_amps = numpy.append(AP_amps, value['AP amplitude'])
                    AP_half_durs = numpy.append(AP_half_durs, value['AP half-duration']) 

            prediction['basal']['long input'][dist].update({  
                                              'AP amplitude' : {'mean': numpy.mean(AP_amps)*mV, 'std' : numpy.std(AP_amps)*mV}, 
                                              'AP half-duration' : {'mean': numpy.mean(AP_half_durs)*ms , 'std' : numpy.std(AP_half_durs)*ms}
                                             })
            prediction_to_json['basal']['long input'][dist].update({  
                                              'AP amplitude' : {'mean': str(numpy.mean(AP_amps)*mV), 'std' : str(numpy.std(AP_amps)*mV)}, 
                                              'AP half-duration' : {'mean': str(numpy.mean(AP_half_durs)*ms) , 'std' : str(numpy.std(AP_half_durs)*ms)}
                                             })

        for freq in frequencies:

            prediction['apical']['train of brief inputs'].update({str(freq) + ' Hz' : collections.OrderedDict()})
            prediction_to_json['apical']['train of brief inputs'].update({str(freq) + ' Hz' : collections.OrderedDict()})
            prediction['basal']['train of brief inputs'].update({str(freq) + ' Hz' : collections.OrderedDict()})
            prediction_to_json['basal']['train of brief inputs'].update({str(freq) + ' Hz' : collections.OrderedDict()})

            for dist in distances_apical:
                AP_amps_fifth_firth_ratio = numpy.array([])
                AP_half_durs_fifth_firth_ratio = numpy.array([])

                prediction['apical']['train of brief inputs'][str(freq) + ' Hz'].update({dist : collections.OrderedDict()})
                prediction_to_json['apical']['train of brief inputs'][str(freq) + ' Hz'].update({dist : collections.OrderedDict()})

                for key, value in features['apical']['train of brief inputs'][str(freq) + ' Hz'].items():
 
                    if value['distance'] >= dist - tolerance_apical and  value['distance'] < dist + tolerance_apical:

                        AP_amps_fifth_firth_ratio = numpy.append(AP_amps_fifth_firth_ratio, value['AP amplitude-AP5/AP1'])
                        AP_half_durs_fifth_firth_ratio = numpy.append(AP_half_durs_fifth_firth_ratio, value['AP half-duration-AP5/AP1']) 

                prediction['apical']['train of brief inputs'][str(freq) + ' Hz'][dist].update({  
                                              'AP amplitude-AP5/AP1' : {'mean': numpy.mean(AP_amps_fifth_firth_ratio)*mV, 'std' : numpy.std(AP_amps_fifth_firth_ratio)*mV}, 
                                              'AP half-duration-AP5/AP1' : {'mean': numpy.mean(AP_half_durs_fifth_firth_ratio)*ms , 'std' : numpy.std(AP_half_durs_fifth_firth_ratio)*ms}
                                             })
                prediction_to_json['apical']['train of brief inputs'][str(freq) + ' Hz'][dist].update({  
                                              'AP amplitude-AP5/AP1' : {'mean': str(numpy.mean(AP_amps_fifth_firth_ratio)*mV), 'std' : str(numpy.std(AP_amps_fifth_firth_ratio)*mV)}, 
                                              'AP half-duration-AP5/AP1' : {'mean': str(numpy.mean(AP_half_durs_fifth_firth_ratio)*ms) , 'std' : str(numpy.std(AP_half_durs_fifth_firth_ratio)*ms)}
                                             })

            for dist in distances_basal:
                AP_amps_fifth_firth_ratio = numpy.array([])
                AP_half_durs_fifth_firth_ratio = numpy.array([])

                prediction['basal']['train of brief inputs'][str(freq) + ' Hz'].update({dist : collections.OrderedDict()})
                prediction_to_json['basal']['train of brief inputs'][str(freq) + ' Hz'].update({dist : collections.OrderedDict()})

                for key, value in features['basal']['train of brief inputs'][str(freq) + ' Hz'].items():
 
                    if value['distance'] >= dist - tolerance_basal and  value['distance'] < dist + tolerance_basal:

                        AP_amps_fifth_firth_ratio = numpy.append(AP_amps_fifth_firth_ratio, value['AP amplitude-AP5/AP1'])
                        AP_half_durs_fifth_firth_ratio = numpy.append(AP_half_durs_fifth_firth_ratio, value['AP half-duration-AP5/AP1']) 

                prediction['basal']['train of brief inputs'][str(freq) + ' Hz'][dist].update({  
                                              'AP amplitude-AP5/AP1' : {'mean': numpy.mean(AP_amps_fifth_firth_ratio)*mV, 'std' : numpy.std(AP_amps_fifth_firth_ratio)*mV}, 
                                              'AP half-duration-AP5/AP1' : {'mean': numpy.mean(AP_half_durs_fifth_firth_ratio)*ms , 'std' : numpy.std(AP_half_durs_fifth_firth_ratio)*ms}
                                             })
                prediction_to_json['basal']['train of brief inputs'][str(freq) + ' Hz'][dist].update({  
                                              'AP amplitude-AP5/AP1' : {'mean': str(numpy.mean(AP_amps_fifth_firth_ratio)*mV), 'std' : str(numpy.std(AP_amps_fifth_firth_ratio)*mV)}, 
                                              'AP half-duration-AP5/AP1' : {'mean': str(numpy.mean(AP_half_durs_fifth_firth_ratio)*ms) , 'std' : str(numpy.std(AP_half_durs_fifth_firth_ratio)*ms)}
                                             })

        file_name_features = self.path_results + 'bAP_CA3_PC_model_features.json'
        file_name_mean_features = self.path_results + 'bAP_CA3_PC_mean_model_features.json'
        json.dump(features_to_json, open(file_name_features , "w"), indent=4)
        json.dump(prediction_to_json, open(file_name_mean_features , "w"), indent=4)

        if self.save_all:
            file_name_features_p = self.path_results + 'bAP_CA3_PC_model_features.p'
            file_name_mean_features_p = self.path_results + 'bAP_CA3_PC_mean_model_features.p'           
            pickle.dump(features, gzip.GzipFile(file_name_features_p, "wb"))
            pickle.dump(prediction, gzip.GzipFile(file_name_mean_features_p, "wb"))

        return features, prediction

    def plot_features(self, features, prediction):

        observation = self.observation

        fig, axs = plt.subplots(3,2, sharex=True)
        plt.subplots_adjust(hspace = 0.5, wspace = 0.4)
        axs[0, 0].errorbar(0, observation['soma']['AP_amplitude']['mean'], yerr = observation['soma']['AP_amplitude']['std'], marker='o', linestyle='none', color='red') 
        axs[1, 0].errorbar(0, observation['soma']['AP_rise_slope']['mean'], yerr = observation['soma']['AP_rise_slope']['std'], marker='o', linestyle='none', color='red')
        axs[2, 0].errorbar(0, observation['soma']['AP_half_duration']['mean'], yerr = observation['soma']['AP_half_duration']['std'], marker='o', linestyle='none', color='red')
        axs[0, 1].errorbar(0, observation['soma']['AP_amplitude']['mean'], yerr = observation['soma']['AP_amplitude']['std'], marker='o', linestyle='none', color='red', label = 'experiment') 
        axs[1, 1].errorbar(0, observation['soma']['AP_rise_slope']['mean'], yerr = observation['soma']['AP_rise_slope']['std'], marker='o', linestyle='none', color='red')
        axs[2, 1].errorbar(0, observation['soma']['AP_half_duration']['mean'], yerr = observation['soma']['AP_half_duration']['std'], marker='o', linestyle='none', color='red')
     
        axs[0, 0].plot(0, features['soma']['AP_amplitude'], marker='o', linestyle='none', color='black') 
        axs[1, 0].plot(0, features['soma']['AP_rise_slope'], marker='o', linestyle='none', color='black')
        axs[2, 0].plot(0, features['soma']['AP_half_duration'], marker='o', linestyle='none', color='black')
        axs[0, 1].plot(0, features['soma']['AP_amplitude'], marker='o', linestyle='none', color='black', label= 'soma model') 
        axs[1, 1].plot(0, features['soma']['AP_rise_slope'], marker='o', linestyle='none', color='black')
        axs[2, 1].plot(0, features['soma']['AP_half_duration'], marker='o', linestyle='none', color='black')

        for key, value in observation['apical'].items():
            axs[0, 0].errorbar(int(key), value['AP1_amplitude']['mean'], yerr = value['AP1_amplitude']['std'], marker='o', linestyle='none', color='red') 
            axs[1, 0].errorbar(int(key), value['AP1_rise_slope']['mean'], yerr = value['AP1_rise_slope']['std'], marker='o', linestyle='none', color='red')
            axs[2, 0].errorbar(int(key), value['AP1_half_duration']['mean'], yerr = value['AP1_half_duration']['std'], marker='o', linestyle='none', color='red')
            axs[0, 1].errorbar(int(key), value['APlast_amplitude']['mean'], yerr = value['APlast_amplitude']['std'], marker='o', linestyle='none', color='red') 
            axs[1, 1].errorbar(int(key), value['APlast_rise_slope']['mean'], yerr = value['APlast_rise_slope']['std'], marker='o', linestyle='none', color='red')
            axs[2, 1].errorbar(int(key), value['APlast_half_duration']['mean'], yerr = value['APlast_half_duration']['std'], marker='o', linestyle='none', color='red')  
        for key, value in observation['basal'].items():
            axs[0, 0].errorbar(int(key)*-1.0 , value['AP1_amplitude']['mean'], yerr = value['AP1_amplitude']['std'], marker='o', linestyle='none', color='red') 
            axs[1, 0].errorbar(int(key)*-1.0, value['AP1_rise_slope']['mean'], yerr = value['AP1_rise_slope']['std'], marker='o', linestyle='none', color='red')
            axs[2, 0].errorbar(int(key)*-1.0, value['AP1_half_duration']['mean'], yerr = value['AP1_half_duration']['std'], marker='o', linestyle='none', color='red')
            axs[0, 1].errorbar(int(key)*-1.0, value['APlast_amplitude']['mean'], yerr = value['APlast_amplitude']['std'], marker='o', linestyle='none', color='red') 
            axs[1, 1].errorbar(int(key)*-1.0, value['APlast_rise_slope']['mean'], yerr = value['APlast_rise_slope']['std'], marker='o', linestyle='none', color='red')
            axs[2, 1].errorbar(int(key)*-1.0, value['APlast_half_duration']['mean'], yerr = value['APlast_half_duration']['std'], marker='o', linestyle='none', color='red')  

        i=0
        for key, value in features['apical'].items():
            axs[0, 0].plot(value['distance'], value['AP1_amplitude'], marker='o', linestyle='none', color='blue') 
            axs[1, 0].plot(value['distance'], value['AP1_rise_slope'], marker='o', linestyle='none', color='blue')
            axs[2, 0].plot(value['distance'], value['AP1_half_duration'], marker='o', linestyle='none', color='blue')
            if i==0:
                axs[0, 1].plot(value['distance'], value['APlast_amplitude'], marker='o', linestyle='none', color='blue', label='model') 
            else:
                axs[0, 1].plot(value['distance'], value['APlast_amplitude'], marker='o', linestyle='none', color='blue')
            axs[1, 1].plot(value['distance'], value['APlast_rise_slope'], marker='o', linestyle='none', color='blue')
            axs[2, 1].plot(value['distance'], value['APlast_half_duration'], marker='o', linestyle='none', color='blue') 
            i+=1

        for key, value in features['basal'].items():
            axs[0, 0].plot(value['distance']*-1.0 , value['AP1_amplitude'], marker='o', linestyle='none', color='blue') 
            axs[1, 0].plot(value['distance']*-1.0, value['AP1_rise_slope'], marker='o', linestyle='none', color='blue')
            axs[2, 0].plot(value['distance']*-1.0, value['AP1_half_duration'], marker='o', linestyle='none', color='blue')
            axs[0, 1].plot(value['distance']*-1.0, value['APlast_amplitude'], marker='o', linestyle='none', color='blue') 
            axs[1, 1].plot(value['distance']*-1.0, value['APlast_rise_slope'], marker='o', linestyle='none', color='blue')
            axs[2, 1].plot(value['distance']*-1.0, value['APlast_half_duration'], marker='o', linestyle='none', color='blue') 
        axs[0, 0].set_ylabel('AP amplitude\n (mV)') 
        axs[1, 0].set_ylabel('AP rise slope\n (V/ms)') 
        axs[2, 0].set_ylabel('AP half-duration\n (ms)') 
        axs[2, 0].set_xlabel('Distance (um)') 
        axs[2, 1].set_xlabel('Distance (um)') 
        axs[0, 0].set_title('First AP') 
        axs[0, 1].set_title('Last AP')
        fig.suptitle('positive distance: apical dendrites, negative distance: basal dendrites')
        lgd=axs[0,1].legend(bbox_to_anchor=(1.0, 1.0), loc = 'upper left')
        if self.save_all:
            plt.savefig(self.path_figs + 'bAP_BC_features'+ '.pdf', dpi=600, bbox_extra_artists=(lgd,), bbox_inches='tight')


       # plot prediction

        fig2, axs2 = plt.subplots(3,2, sharex=True)
        plt.subplots_adjust(hspace = 0.5, wspace = 0.4)
        axs2[0, 0].errorbar(0, observation['soma']['AP_amplitude']['mean'], yerr = observation['soma']['AP_amplitude']['std'], marker='o', linestyle='none', color='red') 
        axs2[1, 0].errorbar(0, observation['soma']['AP_rise_slope']['mean'], yerr = observation['soma']['AP_rise_slope']['std'], marker='o', linestyle='none', color='red')
        axs2[2, 0].errorbar(0, observation['soma']['AP_half_duration']['mean'], yerr = observation['soma']['AP_half_duration']['std'], marker='o', linestyle='none', color='red')
        axs2[0, 1].errorbar(0, observation['soma']['AP_amplitude']['mean'], yerr = observation['soma']['AP_amplitude']['std'], marker='o', linestyle='none', color='red', label = 'experiment') 
        axs2[1, 1].errorbar(0, observation['soma']['AP_rise_slope']['mean'], yerr = observation['soma']['AP_rise_slope']['std'], marker='o', linestyle='none', color='red')
        axs[2, 1].errorbar(0, observation['soma']['AP_half_duration']['mean'], yerr = observation['soma']['AP_half_duration']['std'], marker='o', linestyle='none', color='red')
     
        axs2[0, 0].plot(0, features['soma']['AP_amplitude'], marker='o', linestyle='none', color='black') 
        axs2[1, 0].plot(0, features['soma']['AP_rise_slope'], marker='o', linestyle='none', color='black')
        axs2[2, 0].plot(0, features['soma']['AP_half_duration'], marker='o', linestyle='none', color='black')
        axs2[0, 1].plot(0, features['soma']['AP_amplitude'], marker='o', linestyle='none', color='black', label= 'soma model') 
        axs2[1, 1].plot(0, features['soma']['AP_rise_slope'], marker='o', linestyle='none', color='black')
        axs2[2, 1].plot(0, features['soma']['AP_half_duration'], marker='o', linestyle='none', color='black')

        for key, value in observation['apical'].items():
            axs2[0, 0].errorbar(int(key), value['AP1_amplitude']['mean'], yerr = value['AP1_amplitude']['std'], marker='o', linestyle='none', color='red') 
            axs2[1, 0].errorbar(int(key), value['AP1_rise_slope']['mean'], yerr = value['AP1_rise_slope']['std'], marker='o', linestyle='none', color='red')
            axs2[2, 0].errorbar(int(key), value['AP1_half_duration']['mean'], yerr = value['AP1_half_duration']['std'], marker='o', linestyle='none', color='red')
            axs2[0, 1].errorbar(int(key), value['APlast_amplitude']['mean'], yerr = value['APlast_amplitude']['std'], marker='o', linestyle='none', color='red') 
            axs2[1, 1].errorbar(int(key), value['APlast_rise_slope']['mean'], yerr = value['APlast_rise_slope']['std'], marker='o', linestyle='none', color='red')
            axs2[2, 1].errorbar(int(key), value['APlast_half_duration']['mean'], yerr = value['APlast_half_duration']['std'], marker='o', linestyle='none', color='red')  
        for key, value in observation['basal'].items():
            axs2[0, 0].errorbar(int(key)*-1.0 , value['AP1_amplitude']['mean'], yerr = value['AP1_amplitude']['std'], marker='o', linestyle='none', color='red') 
            axs2[1, 0].errorbar(int(key)*-1.0, value['AP1_rise_slope']['mean'], yerr = value['AP1_rise_slope']['std'], marker='o', linestyle='none', color='red')
            axs2[2, 0].errorbar(int(key)*-1.0, value['AP1_half_duration']['mean'], yerr = value['AP1_half_duration']['std'], marker='o', linestyle='none', color='red')
            axs2[0, 1].errorbar(int(key)*-1.0, value['APlast_amplitude']['mean'], yerr = value['APlast_amplitude']['std'], marker='o', linestyle='none', color='red') 
            axs2[1, 1].errorbar(int(key)*-1.0, value['APlast_rise_slope']['mean'], yerr = value['APlast_rise_slope']['std'], marker='o', linestyle='none', color='red')
            axs2[2, 1].errorbar(int(key)*-1.0, value['APlast_half_duration']['mean'], yerr = value['APlast_half_duration']['std'], marker='o', linestyle='none', color='red')  

        i=0
        for key, value in prediction['apical'].items():
            axs2[0, 0].errorbar(int(key), value['AP1_amplitude']['mean'], yerr = value['AP1_amplitude']['std'], marker='o', linestyle='none', color='blue') 
            axs2[1, 0].errorbar(int(key), value['AP1_rise_slope']['mean'], yerr = value['AP1_rise_slope']['std'], marker='o', linestyle='none', color='blue')
            axs2[2, 0].errorbar(int(key), value['AP1_half_duration']['mean'], yerr = value['AP1_half_duration']['std'], marker='o', linestyle='none', color='blue')
            if i==0:
                axs2[0, 1].errorbar(int(key), value['APlast_amplitude']['mean'], yerr = value['APlast_amplitude']['std'], marker='o', linestyle='none', color='blue', label = 'model')
            else:
                axs2[0, 1].errorbar(int(key), value['APlast_amplitude']['mean'], yerr = value['APlast_amplitude']['std'], marker='o', linestyle='none', color='blue', label = 'model') 
            axs2[1, 1].errorbar(int(key), value['APlast_rise_slope']['mean'], yerr = value['APlast_rise_slope']['std'], marker='o', linestyle='none', color='blue')
            axs2[2, 1].errorbar(int(key), value['APlast_half_duration']['mean'], yerr = value['APlast_half_duration']['std'], marker='o', linestyle='none', color='blue') 
            i+=1
 
        for key, value in prediction['basal'].items():
            axs2[0, 0].errorbar(int(key)*-1.0 , value['AP1_amplitude']['mean'], yerr = value['AP1_amplitude']['std'], marker='o', linestyle='none', color='blue') 
            axs2[1, 0].errorbar(int(key)*-1.0, value['AP1_rise_slope']['mean'], yerr = value['AP1_rise_slope']['std'], marker='o', linestyle='none', color='blue')
            axs2[2, 0].errorbar(int(key)*-1.0, value['AP1_half_duration']['mean'], yerr = value['AP1_half_duration']['std'], marker='o', linestyle='none', color='blue')
            axs2[0, 1].errorbar(int(key)*-1.0, value['APlast_amplitude']['mean'], yerr = value['APlast_amplitude']['std'], marker='o', linestyle='none', color='blue') 
            axs2[1, 1].errorbar(int(key)*-1.0, value['APlast_rise_slope']['mean'], yerr = value['APlast_rise_slope']['std'], marker='o', linestyle='none', color='blue')
            axs2[2, 1].errorbar(int(key)*-1.0, value['APlast_half_duration']['mean'], yerr = value['APlast_half_duration']['std'], marker='o', linestyle='none', color='blue')   
        axs2[0, 0].set_ylabel('AP amplitude\n (mV)') 
        axs2[1, 0].set_ylabel('AP rise slope\n (V/ms)') 
        axs2[2, 0].set_ylabel('AP half-duration\n (ms)') 
        axs2[2, 0].set_xlabel('Distance (um)') 
        axs2[2, 1].set_xlabel('Distance (um)') 
        axs2[0, 0].set_title('First AP') 
        axs2[0, 1].set_title('Last AP')
        fig2.suptitle('positive distance: apical dendrites, negative distance: basal dendrites')
        lgd=axs[0,1].legend(bbox_to_anchor=(1.0, 1.0), loc = 'upper left')
        if self.save_all:
            plt.savefig(self.path_figs + 'bAP_BC_mean_features'+ '.pdf', dpi=600, bbox_extra_artists=(lgd,), bbox_inches='tight') 
       
    def plot_errors(self, errors):

        n_dist_apic = len(list(errors['apical'].keys()))
        n_dist_bas = len(list(errors['basal'].keys()))
        n_subplots = 1+n_dist_apic+n_dist_bas
        fig, axs = plt.subplots(int(numpy.ceil(n_subplots/2)), 2)
        axs = axs.flatten()
        plt.subplots_adjust(hspace = 0.7, wspace = 1.1)

        i = 0   
        for key, value in errors['apical'].items():
           err =[]
           ticks =[]
           for k, v in value.items():
               err.append(v)
               ticks.append(k) 
           y = list(range(len(ticks)))
           axs[i].plot(err, y, 'o') 
           axs[i].set_yticks(y)
           axs[i].set_yticklabels(ticks)
           axs[i].set_title('Apical dendrites - ' + str(key) + ' um')
           i+=1

        for key, value in errors['basal'].items():
           err =[]
           ticks =[]
           for k, v in value.items():
               err.append(v)
               ticks.append(k) 
           y = list(range(len(ticks)))
           axs[i].plot(err, y, 'o') 
           axs[i].set_yticks(y) 
           axs[i].set_yticklabels(ticks)
           axs[i].set_title('Basal dendrites - ' + str(key) + ' um')
           i+=1

        err =[]
        ticks =[]

        for k,v in errors['soma'].items():
            err.append(v)
            ticks.append(k)
        y = list(range(len(ticks))) 
        axs[i].plot(err, y, 'o')
        axs[i].set_yticks(y)
        axs[i].set_yticklabels(ticks)
        axs[i].set_title('Soma')
        fig.suptitle('Feature errors')
        if self.save_all:
            plt.savefig(self.path_figs + 'bAP_BC_feature_errors'+ '.pdf', dpi=600, bbox_inches='tight') 

    def plot_traces(self, model, traces_soma_and_apical, traces_soma_and_basal, apical_locations_distances, basal_locations_distances):
        # TODO: somehow sort the traces by distance 

        if self.base_directory:
            self.path_figs = self.base_directory + 'figs/' + 'backpropagating_AP_CA3_PC/' + model.name + '/'
        else:
            self.path_figs = model.base_directory + 'figs/' + 'backpropagating_AP_CA3_PC/'


        try:
            if not os.path.exists(self.path_figs) and self.save_all:
                os.makedirs(self.path_figs)
        except OSError as e:
            if e.errno != 17:
                raise
            pass

        print("The figures are saved in the directory: ", self.path_figs)

        fig1, axs1 = plt.subplots(len(list(traces_soma_and_apical['train of brief inputs'].keys())), 1, sharex = True, sharey = True)
        plt.subplots_adjust(hspace = 0.4)

        for i, freq in enumerate(list(traces_soma_and_apical['train of brief inputs'].keys())):
            axs1[i].plot(traces_soma_and_apical['train of brief inputs'][freq]['T'],traces_soma_and_apical['train of brief inputs'][freq]['v_stim'], 'r', label = 'soma')
            for key, value in traces_soma_and_apical['train of brief inputs'][freq]['v_rec'].items():
                 axs1[i].plot(traces_soma_and_apical['train of brief inputs'][freq]['T'],traces_soma_and_apical['train of brief inputs'][freq]['v_rec'][key], label = key[0]+'('+str(key[1])+') at '+str(apical_locations_distances[key])+' um')
            axs1[i].set_ylabel('membrane\n potential (mV)')
            axs1[i].text(0.95,0.9, str(freq) + ' Hz', horizontalalignment='right', verticalalignment='top', transform=axs1[i].transAxes)
        axs1[-1].set_xlabel('time (ms)')
        fig1.suptitle('Brief current pulses to soma\n recorded at apical dendrites')
        lgd=axs1[0].legend(bbox_to_anchor=(1.0, 1.0), loc = 'upper left')
        if self.save_all:
            plt.savefig(self.path_figs + 'traces_pulses_apical'+ '.pdf', dpi=600, bbox_extra_artists=(lgd,), bbox_inches='tight')


        fig2, axs2 = plt.subplots(len(list(traces_soma_and_basal['train of brief inputs'].keys())), 1, sharex = True)
        plt.subplots_adjust(hspace = 0.4)

        for i, freq in enumerate(list(traces_soma_and_basal['train of brief inputs'].keys())):
            axs2[i].plot(traces_soma_and_basal['train of brief inputs'][freq]['T'],traces_soma_and_basal['train of brief inputs'][freq]['v_stim'], 'r', label = 'soma')
            for key, value in traces_soma_and_basal['train of brief inputs'][freq]['v_rec'].items():
                axs2[i].plot(traces_soma_and_basal['train of brief inputs'][freq]['T'],traces_soma_and_basal['train of brief inputs'][freq]['v_rec'][key], label = key[0]+'('+str(key[1])+') at '+str(basal_locations_distances[key])+' um')
            axs2[i].set_ylabel('membrane\n potential (mV)')
            axs2[i].text(0.95,0.9, str(freq) + ' Hz', horizontalalignment='right', verticalalignment='top', transform=axs2[i].transAxes)
        axs2[-1].set_xlabel('time (ms)')
        fig2.suptitle('Brief current pulses to soma\n recorded at basal dendrites')
        lgd=axs2[0].legend(bbox_to_anchor=(1.0, 1.0), loc = 'upper left')
        if self.save_all:
            plt.savefig(self.path_figs + 'traces_pulses_basal'+ '.pdf', dpi=600, bbox_extra_artists=(lgd,), bbox_inches='tight')


        plt.figure()
        plt.plot(traces_soma_and_apical['long input']['T'],traces_soma_and_apical['long input']['v_stim'], 'r', label = 'soma')
        for key, value in traces_soma_and_apical['long input']['v_rec'].items():
            plt.plot(traces_soma_and_apical['long input']['T'],traces_soma_and_apical['long input']['v_rec'][key], label = key[0]+'('+str(key[1])+') at '+str(apical_locations_distances[key])+' um')
        plt.xlabel('time (ms)')
        plt.ylabel('membrane potential (mV)')
        plt.title('Long rheobase current to soma\n recorded at apical dendrites')
        lgd=plt.legend(bbox_to_anchor=(1.0, 1.0), loc = 'upper left')
        if self.save_all:
            plt.savefig(self.path_figs + 'traces_long_rheobase_input_apical'+ '.pdf', dpi=600, bbox_extra_artists=(lgd,), bbox_inches='tight')

        plt.figure()
        plt.plot(traces_soma_and_basal['long input']['T'],traces_soma_and_basal['long input']['v_stim'], 'r', label = 'soma')
        for key, value in traces_soma_and_basal['long input']['v_rec'].items():
            plt.plot(traces_soma_and_basal['long input']['T'],traces_soma_and_basal['long input']['v_rec'][key], label = key[0]+'('+str(key[1])+') at '+str(basal_locations_distances[key])+' um')
        plt.xlabel('time (ms)')
        plt.ylabel('membrane potential (mV)')
        plt.title('Long rheobase current to soma\n recorded at basal dendrites')
        lgd=plt.legend(bbox_to_anchor=(1.0, 1.0), loc = 'upper left')
        if self.save_all:
            plt.savefig(self.path_figs + 'traces_long_rheobase_input_basal'+ '.pdf', dpi=600, bbox_extra_artists=(lgd,), bbox_inches='tight')


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
            self.path_results = self.base_directory + 'results/' + 'backpropagating_AP_CA3_PC/' + model.name + '/'
        else:
            self.path_results = model.base_directory + 'results/' + 'backpropagating_AP_CA3_PC/'

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

        apical_locations, apical_locations_distances = model.get_random_locations_multiproc('apical', self.num_of_apical_dend_locations, self.random_seed, dist_range_apical) # number of random locations , seed
        # apical_locations_distances = sorted(apical_locations_distances, key=apical_locations_distances.get)
        #print dend_locations, actual_distances
        print('Apical dendritic locations to be tested (with their actual distances):', apical_locations_distances)

        basal_locations, basal_locations_distances = model.get_random_locations_multiproc('basal', self.num_of_basal_dend_locations, self.random_seed, dist_range_basal) # number of random locations , seed
        # basal_locations_distances = sorted(basal_locations_distances, key=basal_locations_distances.get)
        #print dend_locations, actual_distances
        print('Basal dendritic locations to be tested (with their actual distances):', basal_locations_distances)

        dend_locations = apical_locations + basal_locations   # so the simulation is run only once, and record from alll the locations at the same time 


        delay = self.config['train of brief inputs']['delay']
        dur_of_pulse = self.config['train of brief inputs']['duration of single pulse']
        amp = self.config['train of brief inputs']['amplitude of pulses']
        num_of_pulses = self.config['train of brief inputs']['number of pulses']
        frequencies = self.config['train of brief inputs']['frequencies']

        max_interval_bw_pulses = 1.0/ min(frequencies) * 1000.0
        dur_of_stim = num_of_pulses * max_interval_bw_pulses

        plt.close('all') #needed to avoid overlapping of saved images when the test is run on multiple models


        pool = multiprocessing.Pool(self.npool, maxtasksperchild=1)
        current_pulses_ = functools.partial(self.current_pulses, model, amp, delay, dur_of_pulse, dur_of_stim, num_of_pulses, "soma", 0.5, dend_locations)
        traces_train = pool.map(current_pulses_ , frequencies, chunksize=1)
        pool.terminate()
        pool.join()
        del pool
        # print(traces_train)
        '''
        traces_all = {'long input' : {},
                      'train of brief inputs' :{}}  
        '''

        traces_soma_and_basal = collections.OrderedDict() 
        traces_soma_and_basal['train of brief inputs'] = collections.OrderedDict()
        traces_soma_and_basal['long input'] = collections.OrderedDict()
        # traces_soma_and_basal['train of brief inputs']['T'] = traces_train['T'] 

        traces_soma_and_apical = collections.OrderedDict() 
        traces_soma_and_apical['train of brief inputs'] = collections.OrderedDict()
        traces_soma_and_apical['long input'] = collections.OrderedDict()
        # traces_soma_and_apical['train of brief inputs']['T'] = traces_train['T'] 

        for i, freq in enumerate(frequencies):
            traces_soma_and_apical['train of brief inputs'].update({freq : {'v_rec' : {}}}) 
            traces_soma_and_basal['train of brief inputs'].update({freq : {'v_rec' : {}}}) 

            traces_soma_and_apical['train of brief inputs'][freq].update({'v_stim' : traces_train[i]['v_stim']})
            traces_soma_and_basal['train of brief inputs'][freq].update({'v_stim' : traces_train[i]['v_stim']})
            traces_soma_and_apical['train of brief inputs'][freq].update({'T' : traces_train[i]['T']})
            traces_soma_and_basal['train of brief inputs'][freq].update({'T' : traces_train[i]['T']})

            for key, value in traces_train[i]['v_rec'].items():
                if list(key) in apical_locations: 
                    traces_soma_and_apical['train of brief inputs'][freq]['v_rec'].update({key:value})
                if list(key) in basal_locations:
                    traces_soma_and_basal['train of brief inputs'][freq]['v_rec'].update({key:value})

        # print(traces_soma_and_apical)
        # print(traces_soma_and_basal)    

        '''
        TODO
        amplitude = self.find_rheobase()
        ''' 
        amplitude = 0.6
        delay_long = self.config['long input']['delay'] 
        duration_long = self.config['long input']['duration']

        pool = multiprocessing.Pool(1, maxtasksperchild = 1)
        traces = pool.apply(self.long_rheobase_input, args = (model, amplitude, delay_long, duration_long, "soma", 0.5, dend_locations))
        pool.terminate()
        pool.join()
        del pool
        # print(traces)

        traces_soma_and_apical['long input'].update({'v_rec' : {}})
        traces_soma_and_apical['long input'].update({'v_stim' : traces['v_stim']})
        traces_soma_and_apical['long input'].update({'T' : traces['T']})

        traces_soma_and_basal['long input'].update({'v_rec' : {}})
        traces_soma_and_basal['long input'].update({'v_stim' : traces['v_stim']})
        traces_soma_and_basal['long input'].update({'T' : traces['T']})

        for key, value in traces['v_rec'].items():
            if list(key) in apical_locations:
               traces_soma_and_apical['long input']['v_rec'].update({key:value}) 
            if list(key) in basal_locations:
               traces_soma_and_basal['long input']['v_rec'].update({key:value})
  

        filepath = self.path_results + self.test_log_filename
        self.logFile = open(filepath, 'w') # if it is opened before multiprocessing, the multiporeccing won't work under python3

        self.logFile.write('Apical dendritic locations to be tested (with their actual distances):\n'+ str(apical_locations_distances)+'\n')
        self.logFile.write('Basal dendritic locations to be tested (with their actual distances):\n'+ str(basal_locations_distances)+'\n')
        self.logFile.write("---------------------------------------------------------------------------------------------------\n")


        self.plot_traces(model, traces_soma_and_apical, traces_soma_and_basal, apical_locations_distances, basal_locations_distances)


        efel_features_somatic, efel_features_apical, efel_features_basal = self.extract_features_by_eFEL(traces_soma_and_apical, traces_soma_and_basal, delay, dur_of_stim, delay_long, duration_long, frequencies)

        # print(efel_features_somatic, efel_features_apical, efel_features_basal)

        time_indices_befor_and_after_somatic_AP = self.get_time_indices_befor_and_after_somatic_AP(efel_features_somatic, traces_soma_and_apical, frequencies)

        self.plot_AP1_AP5(time_indices_befor_and_after_somatic_AP, apical_locations_distances, basal_locations_distances, traces_soma_and_apical, traces_soma_and_basal, frequencies)


        features, prediction = self.extract_prediction_features(efel_features_somatic, efel_features_apical, efel_features_basal, apical_locations_distances, basal_locations_distances, distances_apical, tolerance_apical, distances_basal, tolerance_basal, frequencies)
        """ Till this point  it works . Except that the determination of rheobase is missing"""

        self.plot_features(features, prediction)

        efel.reset()

        return prediction

    def compute_score(self, observation, prediction, verbose=False):
        """Implementation of sciunit.Test.score_prediction."""


        score_avg, errors= scores.ZScore_backpropagatingAP_BasketCell.compute(observation,prediction)

        file_name=self.path_results+'bAP_CA3_PC_errors.json'

        json.dump(errors, open(file_name, "w"), indent=4)

        file_name_s=self.path_results+'bAP_scores.json'


        self.plot_errors(errors)

        if self.show_plot:
            plt.show()

            
        score_json= {'Z_score_avg' : score_avg}


        file_name_score = self.path_results + 'bAP_CA3_PC_final_score.json'
        json.dump(score_json, open(file_name_score, "w"), indent=4)


        score=scores.ZScore_backpropagatingAP_BasketCell(score_avg)

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
