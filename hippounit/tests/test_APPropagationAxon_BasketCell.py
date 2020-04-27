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


class APPropagationAxonTest_BasketCell(Test):
    """Tests efficacy and shape of propagating action potentials to the axons from the soma."""

    def __init__(self, config = {},
                observation = {},
                name="APPropagationAxonTest" ,
                force_run=False,
                base_directory= None,
                show_plot=True,
                save_all = True,
                num_of_axonal_locations = 15,
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

        self.num_of_axonal_locations = num_of_axonal_locations

        self.config = config

        description = "Tests efficacy and shape of propagating action potentials to the axon from the soma."

    score_type = scores.ZScore_APPropagationAxon_BasketCell

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
            self.path_temp_data = self.base_directory + 'temp_data/' + 'AP_propagation_axon_BC/' + model.name + '/'
        else:
            self.path_temp_data = model.base_directory + 'temp_data/' + 'AP_propagation_axon_BC/'


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

    def extract_features_by_eFEL(self, traces_soma_axon, delay, duration):


        # soma
        trace_soma = {}
        traces_soma=[]
        trace_soma['T'] = traces_soma_axon['T']
        trace_soma['V'] = traces_soma_axon['v_stim']
        trace_soma['stim_start'] = [delay]
        trace_soma['stim_end'] = [delay + duration]
        traces_soma.append(trace_soma)


        traces_results_soma = efel.getFeatureValues(traces_soma, ['AP_amplitude', 'AP_duration_half_width', 'inv_first_ISI','AP_begin_time', 'doublet_ISI'])
   
        #axon
        traces_results_axon = {} 
        for key, value in traces_soma_axon['v_rec'].items():
            trace = {}
            traces_for_efel=[]
            trace['T'] = traces_soma_axon['T']
            trace['V'] = value
            trace['stim_start'] = [delay]
            trace['stim_end'] = [delay + duration]
            traces_for_efel.append(trace)


            traces_results = efel.getFeatureValues(traces_for_efel, ['AP_amplitude', 'AP_duration_half_width'])
            traces_results_axon[key] = traces_results



        return traces_results_soma, traces_results_axon


    def get_time_indices_befor_and_after_somatic_AP(self, efel_features_somatic, traces_soma_axon):


        soma_AP_begin_time = efel_features_somatic[0]['AP_begin_time']
        #soma_inv_first_ISI = traces_results[0]['inv_first_ISI']
        soma_first_ISI = efel_features_somatic[0]['doublet_ISI'][0]
        #print soma_AP_begin_time[0], soma_AP_begin_time[0]-1
        #print traces_results[0]['inv_first_ISI'], soma_first_ISI
        s_indices_AP1 = numpy.where(traces_soma_axon['T'] >= (soma_AP_begin_time[0]-1.0))
        if 10 < soma_first_ISI:
            plus = 10
        else:
            plus = soma_first_ISI-3
        e_indices_AP1 = numpy.where(traces_soma_axon['T'] >= (soma_AP_begin_time[0]+plus))
        start_index_AP1 = s_indices_AP1[0][0]
        end_index_AP1 = e_indices_AP1[0][0]
        #print start_index_AP1
        #print end_index_AP1

        s_indices_APlast = numpy.where(traces_soma_axon['T'] >= soma_AP_begin_time[-1]-1.0)
        e_indices_APlast = numpy.where(traces_soma_axon['T'] >= soma_AP_begin_time[-1]+10)
        start_index_APlast = s_indices_APlast[0][0]
        end_index_APlast = e_indices_APlast[0][0]
        # print([start_index_AP1, end_index_AP1, start_index_APlast, end_index_APlast])
        return [start_index_AP1, end_index_AP1, start_index_APlast, end_index_APlast]  

    def plot_AP1_APlast(self, time_indices_befor_and_after_somatic_AP, axon_locations_distances, traces_soma_axon):

        start_index_AP1, end_index_AP1, start_index_APlast, end_index_APlast = time_indices_befor_and_after_somatic_AP

        # zoom to first and last AP 
        fig1, axs1 = plt.subplots(1,2)
        plt.subplots_adjust(wspace = 0.4)
        axs1[0].plot(traces_soma_axon['T'],traces_soma_axon['v_stim'], 'r')
        axs1[1].plot(traces_soma_axon['T'],traces_soma_axon['v_stim'], 'r', label = 'soma')
        for key, value in traces_soma_axon['v_rec'].items():
            axs1[0].plot(traces_soma_axon['T'],traces_soma_axon['v_rec'][key])
            axs1[1].plot(traces_soma_axon['T'],traces_soma_axon['v_rec'][key], label = key[0]+'('+str(key[1])+') at '+str(axon_locations_distances[key])+' um')
        axs1[0].set_xlabel('time (ms)')
        axs1[0].set_ylabel('membrane potential (mV)')
        axs1[0].set_title('First AP')
        axs1[0].set_xlim(traces_soma_axon['T'][start_index_AP1], traces_soma_axon['T'][end_index_AP1])
        axs1[1].set_xlabel('time (ms)')
        axs1[1].set_ylabel('membrane potential (mV)')
        axs1[1].set_title('Last AP')
        axs1[1].set_xlim(traces_soma_axon['T'][start_index_APlast], traces_soma_axon['T'][end_index_APlast])
        lgd=axs1[1].legend(bbox_to_anchor=(1.0, 1.0), loc = 'upper left')
        if self.save_all:
            plt.savefig(self.path_figs + 'First_and_last_AP'+ '.pdf', dpi=600, bbox_extra_artists=(lgd,), bbox_inches='tight')



    def extract_prediction_features(self, efel_features_somatic, efel_features_axon, axon_locations_distances, distances, tolerance):


        features = {'soma' : { 'AP1_amplitude' : numpy.mean(efel_features_somatic[0]['AP_amplitude'][0])*mV,
                              'AP1_half_duration' : numpy.mean(efel_features_somatic[0]['AP_duration_half_width'][0])*ms,
                              'APlast_amplitude' : numpy.mean(efel_features_somatic[0]['AP_amplitude'][-1])*mV,
                              'APlast_half_duration' : numpy.mean(efel_features_somatic[0]['AP_duration_half_width'][-1])*ms},
                    'axon' : collections.OrderedDict()
                    }  

        features_to_json = {'soma' : { 'AP1_amplitude' : str(numpy.mean(efel_features_somatic[0]['AP_amplitude'][0])*mV),
                              'AP1_half_duration' : str(numpy.mean(efel_features_somatic[0]['AP_duration_half_width'][0])*ms),
                              'APlast_amplitude' : str(numpy.mean(efel_features_somatic[0]['AP_amplitude'][-1])*mV),
                              'APlast_half_duration' : str(numpy.mean(efel_features_somatic[0]['AP_duration_half_width'][-1])*ms)},  
                    'axon' :collections.OrderedDict()
                    } 

        for key, value in efel_features_axon.items():
            features['axon'].update({key :{ 'distance' : axon_locations_distances[key], 
                                              'AP1_amplitude' : value[0]['AP_amplitude'][0]*mV,
                                              'AP1_half_duration' : value[0]['AP_duration_half_width'][0]*ms,
                                              'APlast_amplitude' : value[0]['AP_amplitude'][-1]*mV,
                                              'APlast_half_duration' : value[0]['AP_duration_half_width'][-1]*ms
                                             } 
                                        })

            features_to_json['axon'].update({str(key ):{ 'distance' : axon_locations_distances[key], 
                                              'AP1_amplitude' : str(value[0]['AP_amplitude'][0]*mV),
                                              'AP1_half_duration' : str(value[0]['AP_duration_half_width'][0]*ms),
                                              'APlast_amplitude' : str(value[0]['AP_amplitude'][-1]*mV),
                                              'APlast_half_duration' : str(value[0]['AP_duration_half_width'][-1]*ms)
                                             } 
                                        })


        prediction = {'soma' : features['soma'],
                    'axon' :{}
                    }  

        prediction_to_json = {'soma' : features_to_json['soma'],
                              'axon' :collections.OrderedDict()
                              } 

        for dist in distances:
            AP1_amps = numpy.array([])
            AP1_half_durs = numpy.array([])
            APlast_amps = numpy.array([])
            APlast_half_durs = numpy.array([])
            prediction['axon'].update({dist : collections.OrderedDict()})
            prediction_to_json['axon'].update({dist : collections.OrderedDict()})

            for key, value in features['axon'].items():
 
                if value['distance'] >= dist - tolerance and  value['distance'] < dist + tolerance:
      

                    AP1_amps = numpy.append(AP1_amps, value['AP1_amplitude'])
                    AP1_half_durs = numpy.append(AP1_half_durs, value['AP1_half_duration']) 
                    APlast_amps = numpy.append(AP1_amps, value['APlast_amplitude'])
                    APlast_half_durs = numpy.append(AP1_half_durs, value['APlast_half_duration']) 
            prediction['axon'][dist].update({  
                                              'AP1_amplitude' : {'mean': numpy.mean(AP1_amps)*mV, 'std' : numpy.std(AP1_amps)*mV}, 
                                              'AP1_half_duration' : {'mean': numpy.mean(AP1_half_durs)*ms , 'std' : numpy.std(AP1_half_durs)*ms},
                                              'APlast_amplitude' : {'mean': numpy.mean(APlast_amps)*mV, 'std' : numpy.std(APlast_amps)*mV},
                                              'APlast_half_duration' : {'mean': numpy.mean(APlast_half_durs)*ms, 'std' : numpy.std(APlast_half_durs)*ms}
                                            })

            prediction_to_json['axon'][dist].update({  
                                              'AP1_amplitude' : {'mean': str(numpy.mean(AP1_amps)*mV), 'std' : str(numpy.std(AP1_amps)*mV)}, 
                                              'AP1_half_duration' : {'mean': str(numpy.mean(AP1_half_durs)*ms), 'std' : str(numpy.std(AP1_half_durs)*ms)},
                                              'APlast_amplitude' : {'mean': str(numpy.mean(APlast_amps)*mV), 'std' : str(numpy.std(APlast_amps)*mV)},
                                              'APlast_half_duration' : {'mean': str(numpy.mean(APlast_half_durs)*ms), 'std' : str(numpy.std(APlast_half_durs)*ms)}
                                                  })

        file_name_features = self.path_results + 'AP_prop_axon_BasketCell_model_features.json'
        file_name_mean_features = self.path_results + 'AP_prop_axon_BasketCell_mean_model_features.json'
        json.dump(features_to_json, open(file_name_features , "w"), indent=4)
        json.dump(prediction_to_json, open(file_name_mean_features , "w"), indent=4)

        if self.save_all:
            file_name_features_p = self.path_results + 'AP_prop_axon_BasketCell_model_features.p'
            file_name_mean_features_p = self.path_results + 'AP_prop_axon_BasketCell_mean_model_features.p'           
            pickle.dump(features, gzip.GzipFile(file_name_features_p, "wb"))
            pickle.dump(prediction, gzip.GzipFile(file_name_mean_features_p, "wb"))

        return features, prediction

    def plot_features(self, features, prediction):

        observation = self.observation

        fig, axs = plt.subplots(2,2, sharex=True)
        plt.subplots_adjust(hspace = 0.5, wspace = 0.4)
        axs[0, 0].errorbar(0, observation['soma']['AP1_amplitude']['mean'], yerr = observation['soma']['AP1_amplitude']['std'], marker='o', linestyle='none', color='red') 
        axs[1, 0].errorbar(0, observation['soma']['AP1_half_duration']['mean'], yerr = observation['soma']['AP1_half_duration']['std'], marker='o', linestyle='none', color='red')
        axs[0, 1].errorbar(0, observation['soma']['APlast_amplitude']['mean'], yerr = observation['soma']['APlast_amplitude']['std'], marker='o', linestyle='none', color='red', label = 'experiment') 
        axs[1, 1].errorbar(0, observation['soma']['APlast_half_duration']['mean'], yerr = observation['soma']['APlast_half_duration']['std'], marker='o', linestyle='none', color='red')
     
        axs[0, 0].plot(0, features['soma']['AP1_amplitude'], marker='o', linestyle='none', color='black') 
        axs[1, 0].plot(0, features['soma']['AP1_half_duration'], marker='o', linestyle='none', color='black')
        axs[0, 1].plot(0, features['soma']['APlast_amplitude'], marker='o', linestyle='none', color='black', label= 'soma model') 
        axs[1, 1].plot(0, features['soma']['APlast_half_duration'], marker='o', linestyle='none', color='black')

        for key, value in observation['axon'].items():
            axs[0, 0].errorbar(int(key), value['AP1_amplitude']['mean'], yerr = value['AP1_amplitude']['std'], marker='o', linestyle='none', color='red') 
            axs[1, 0].errorbar(int(key), value['AP1_half_duration']['mean'], yerr = value['AP1_half_duration']['std'], marker='o', linestyle='none', color='red')
            axs[0, 1].errorbar(int(key), value['APlast_amplitude']['mean'], yerr = value['APlast_amplitude']['std'], marker='o', linestyle='none', color='red') 
            axs[1, 1].errorbar(int(key), value['APlast_half_duration']['mean'], yerr = value['APlast_half_duration']['std'], marker='o', linestyle='none', color='red')  

        i=0
        for key, value in features['axon'].items():
            axs[0, 0].plot(value['distance'], value['AP1_amplitude'], marker='o', linestyle='none', color='blue') 
            axs[1, 0].plot(value['distance'], value['AP1_half_duration'], marker='o', linestyle='none', color='blue')
            if i==0:
                axs[0, 1].plot(value['distance'], value['APlast_amplitude'], marker='o', linestyle='none', color='blue', label='model') 
            else:
                axs[0, 1].plot(value['distance'], value['APlast_amplitude'], marker='o', linestyle='none', color='blue')
            axs[1, 1].plot(value['distance'], value['APlast_half_duration'], marker='o', linestyle='none', color='blue') 
            i+=1

        axs[0, 0].set_ylabel('AP amplitude\n (mV)') 
        axs[1, 0].set_ylabel('AP half-duration\n (ms)') 
        axs[1, 0].set_xlabel('Distance (um)') 
        axs[1, 1].set_xlabel('Distance (um)') 
        axs[0, 0].set_title('First AP') 
        axs[0, 1].set_title('Last AP')
        fig.suptitle('Feature values')
        lgd=axs[0,1].legend(bbox_to_anchor=(1.0, 1.0), loc = 'upper left')
        if self.save_all:
            plt.savefig(self.path_figs + 'AP_prop_axon_BC_features'+ '.pdf', dpi=600, bbox_extra_artists=(lgd,), bbox_inches='tight')


       # plot prediction

        fig2, axs2 = plt.subplots(2,2, sharex=True)
        plt.subplots_adjust(hspace = 0.5, wspace = 0.4)
        axs2[0, 0].errorbar(0, observation['soma']['AP1_amplitude']['mean'], yerr = observation['soma']['AP1_amplitude']['std'], marker='o', linestyle='none', color='red') 
        axs2[1, 0].errorbar(0, observation['soma']['AP1_half_duration']['mean'], yerr = observation['soma']['AP1_half_duration']['std'], marker='o', linestyle='none', color='red')
        axs2[0, 1].errorbar(0, observation['soma']['APlast_amplitude']['mean'], yerr = observation['soma']['APlast_amplitude']['std'], marker='o', linestyle='none', color='red', label = 'experiment') 
        axs[1, 1].errorbar(0, observation['soma']['APlast_half_duration']['mean'], yerr = observation['soma']['APlast_half_duration']['std'], marker='o', linestyle='none', color='red')
     
        axs2[0, 0].plot(0, features['soma']['AP1_amplitude'], marker='o', linestyle='none', color='black') 
        axs2[1, 0].plot(0, features['soma']['AP1_half_duration'], marker='o', linestyle='none', color='black')
        axs2[0, 1].plot(0, features['soma']['APlast_amplitude'], marker='o', linestyle='none', color='black', label= 'soma model') 
        axs2[1, 1].plot(0, features['soma']['APlast_half_duration'], marker='o', linestyle='none', color='black')

        for key, value in observation['axon'].items():
            axs2[0, 0].errorbar(int(key), value['AP1_amplitude']['mean'], yerr = value['AP1_amplitude']['std'], marker='o', linestyle='none', color='red') 
            axs2[1, 0].errorbar(int(key), value['AP1_half_duration']['mean'], yerr = value['AP1_half_duration']['std'], marker='o', linestyle='none', color='red')
            axs2[0, 1].errorbar(int(key), value['APlast_amplitude']['mean'], yerr = value['APlast_amplitude']['std'], marker='o', linestyle='none', color='red') 
            axs2[1, 1].errorbar(int(key), value['APlast_half_duration']['mean'], yerr = value['APlast_half_duration']['std'], marker='o', linestyle='none', color='red')   

        i=0
        for key, value in prediction['axon'].items():
            axs2[0, 0].errorbar(int(key), value['AP1_amplitude']['mean'], yerr = value['AP1_amplitude']['std'], marker='o', linestyle='none', color='blue') 
            axs2[1, 0].errorbar(int(key), value['AP1_half_duration']['mean'], yerr = value['AP1_half_duration']['std'], marker='o', linestyle='none', color='blue')
            if i==0:
                axs2[0, 1].errorbar(int(key), value['APlast_amplitude']['mean'], yerr = value['APlast_amplitude']['std'], marker='o', linestyle='none', color='blue', label = 'model')
            else:
                axs2[0, 1].errorbar(int(key), value['APlast_amplitude']['mean'], yerr = value['APlast_amplitude']['std'], marker='o', linestyle='none', color='blue', label = 'model') 
            axs2[1, 1].errorbar(int(key), value['APlast_half_duration']['mean'], yerr = value['APlast_half_duration']['std'], marker='o', linestyle='none', color='blue') 
            i+=1
   
        axs2[0, 0].set_ylabel('AP amplitude\n (mV)') 
        axs2[1, 0].set_ylabel('AP half-duration\n (ms)') 
        axs2[1, 0].set_xlabel('Distance (um)') 
        axs2[1, 1].set_xlabel('Distance (um)') 
        axs2[0, 0].set_title('First AP') 
        axs2[0, 1].set_title('Last AP')
        lgd=axs[0,1].legend(bbox_to_anchor=(1.0, 1.0), loc = 'upper left')
        fig2.suptitle('Mean feature values')
        if self.save_all:
            plt.savefig(self.path_figs + 'AP_prop_axon_mean_features'+ '.pdf', dpi=600, bbox_extra_artists=(lgd,), bbox_inches='tight') 
       
    def plot_errors(self, errors):

        n_subplots = len(list(errors['axon'].keys()))+1
        fig, axs = plt.subplots(int(numpy.ceil(n_subplots/2)), 2)
        axs = axs.flatten()
        plt.subplots_adjust(hspace = 0.7, wspace = 1.1)

        i = 0   
        for key, value in errors['axon'].items():
           err =[]
           ticks =[]
           for k, v in value.items():
               err.append(v)
               ticks.append(k) 
           y = list(range(len(ticks)))
           axs[i].plot(err, y, 'o') 
           axs[i].set_yticks(y)
           axs[i].set_yticklabels(ticks)
           axs[i].set_title('Axon - ' + str(key) + ' um')
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
            plt.savefig(self.path_figs + 'AP_prop_axon_BC_feature_errors'+ '.pdf', dpi=600, bbox_inches='tight') 

    def plot_traces(self, model, traces_soma_axon, axon_locations_distances):
        # TODO: somehow sort the traces by distance 

        if self.base_directory:
            self.path_figs = self.base_directory + 'figs/' + 'AP_propagation_axon_BC/' + model.name + '/'
        else:
            self.path_figs = model.base_directory + 'figs/' + 'AP_propagation_axon_BC/'


        try:
            if not os.path.exists(self.path_figs) and self.save_all:
                os.makedirs(self.path_figs)
        except OSError as e:
            if e.errno != 17:
                raise
            pass

        print("The figures are saved in the directory: ", self.path_figs)

        plt.figure(1)
        plt.plot(traces_soma_axon['T'],traces_soma_axon['v_stim'], 'r', label = 'soma')
        for key, value in traces_soma_axon['v_rec'].items():
            plt.plot(traces_soma_axon['T'],traces_soma_axon['v_rec'][key], label = key[0]+'('+str(key[1])+') at '+str(axon_locations_distances[key])+' um')
        plt.xlabel('time (ms)')
        plt.ylabel('membrane potential (mV)')
        lgd=plt.legend(bbox_to_anchor=(1.0, 1.0), loc = 'upper left')
        if self.save_all:
            plt.savefig(self.path_figs + 'traces'+ '.pdf', dpi=600, bbox_extra_artists=(lgd,), bbox_inches='tight')



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
            self.path_results = self.base_directory + 'results/' + 'AP_propagation_axon_BC/' + model.name + '/'
        else:
            self.path_results = model.base_directory + 'results/' + 'AP_propagation_axon_BC/'

        try:
            if not os.path.exists(self.path_results):
                os.makedirs(self.path_results)
        except OSError as e:
            if e.errno != 17:
                raise
            pass


        global model_name_bAP
        model_name_bAP = model.name

        distances = self.config['recording']['distances']
        tolerance = self.config['recording']['tolerance']
        dist_range = [min(distances) - tolerance, max(distances) + tolerance]


        axon_locations, axon_locations_distances = model.get_random_locations_multiproc('axon', self.num_of_axonal_locations, self.random_seed, dist_range) # number of random locations , seed
        print('Axonal locations to be tested (with their actual distances):', axon_locations_distances)


        traces_soma_axon={}
        delay = self.config['stimulus']['delay']
        duration = self.config['stimulus']['duration']
        amplitude = self.config['stimulus']['amplitude']


        plt.close('all') #needed to avoid overlapping of saved images when the test is run on multiple models


        pool = multiprocessing.Pool(1, maxtasksperchild = 1)
        traces = pool.apply(self.cclamp, args = (model, amplitude, delay, duration, "soma", 0.5, axon_locations))
        # print(traces)

        traces_soma_axon = collections.OrderedDict() 
        traces_soma_axon['T'] = traces['T'] 
        traces_soma_axon['v_stim'] = traces['v_stim']  
        traces_soma_axon['v_rec'] = collections.OrderedDict()


        for key, value in traces['v_rec'].items():
            traces_soma_axon['v_rec'].update({key:value}) 
       

        filepath = self.path_results + self.test_log_filename
        self.logFile = open(filepath, 'w') # if it is opened before multiprocessing, the multiporeccing won't work under python3

        self.logFile.write('Axonal locations to be tested (with their actual distances):\n'+ str(axon_locations_distances)+'\n')
        self.logFile.write("---------------------------------------------------------------------------------------------------\n")


        self.plot_traces(model, traces_soma_axon, axon_locations_distances)


        efel_features_somatic, efel_features_axon = self.extract_features_by_eFEL(traces_soma_axon, delay, duration)

        # print(efel_features_somatic, efel_features_apical, efel_features_basal)

        time_indices_befor_and_after_somatic_AP = self.get_time_indices_befor_and_after_somatic_AP(efel_features_somatic, traces_soma_axon)

        self.plot_AP1_APlast(time_indices_befor_and_after_somatic_AP, axon_locations_distances, traces_soma_axon)

        features, prediction = self.extract_prediction_features(efel_features_somatic, efel_features_axon, axon_locations_distances, distances, tolerance)


        self.plot_features(features, prediction)

        efel.reset()

        return prediction

    def compute_score(self, observation, prediction, verbose=False):
        """Implementation of sciunit.Test.score_prediction."""


        score_avg, errors= scores.ZScore_APPropagationAxon_BasketCell.compute(observation,prediction)

        file_name=self.path_results+'AP_prop_axon_BC_errors.json'

        json.dump(errors, open(file_name, "w"), indent=4)

        file_name_s=self.path_results+'AP_prop_axon_scores.json'


        self.plot_errors(errors)

        if self.show_plot:
            plt.show()

            
        score_json= {'Z_score_avg' : score_avg}


        file_name_score = self.path_results + 'AP_prop_axon_BC_final_score.json'
        json.dump(score_json, open(file_name_score, "w"), indent=4)


        score=scores.ZScore_APPropagationAxon_BasketCell(score_avg)

        self.logFile.write(str(score)+'\n')
        self.logFile.write("---------------------------------------------------------------------------------------------------\n")


        self.logFile.close()

        self.logFile = self.path_results + self.test_log_filename

        return score

    def bind_score(self, score, model, observation, prediction):

        score.related_data["figures"] = [self.path_figs + 'First_and_last_AP_basal.pdf', self.path_figs + 'bAP_BC_feature_errors.pdf', self.path_figs + 'First_and_last_AP_apical.pdf', self.path_figs + 'traces_basal.pdf', self.path_figs + 'traces_apical.pdf', self.path_figs + 'bAP_BC_mean_features.pdf', self.path_figs + 'bAP_BC_features.pdf', self.path_results + 'bAP_BasketCell_mean_model_features.json', self.path_results + 'bAP_BasketCell_model_features.p', self.path_results + 'bAP_BC_errors.json', self.path_results + 'bAP_BC_final_score.json', self.path_results + 'bAP_BasketCell_model_features.json', self.path_results + 'bAP_BasketCell_mean_model_features.p', self.path_results + self.test_log_filename]

        score.related_data["results"] = [self.path_results + 'bAP_BasketCell_mean_model_features.json', self.path_results + 'bAP_BasketCell_model_features.p', self.path_results + 'bAP_BC_errors.json', self.path_results + 'bAP_BC_final_score.json', self.path_results + 'bAP_BasketCell_model_features.json', self.path_results + 'bAP_BasketCell_mean_model_features.p']

        return score
