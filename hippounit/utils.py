from __future__ import print_function
from __future__ import division
# from builtins import str
from builtins import range
import os
import numpy
import sciunit
import hippounit.capabilities as cap
from quantities import ms,mV,Hz
from neuron import h

import multiprocessing
import zipfile
import collections

import collections

import json

import pkg_resources
import sys



class ModelLoader(sciunit.Model,
                 cap.ProvidesGoodObliques,
                 cap.ReceivesSquareCurrent_ProvidesResponse,
                 cap.ReceivesSynapse,
                 cap.ReceivesMultipleSynapses,
                 cap.ReceivesSquareCurrent_ProvidesResponse_MultipleLocations,
                 cap.ProvidesRecordingLocationsOnTrunk,
                 cap.ProvidesRandomDendriticLocations,
                 cap.ReceivesEPSCstim,
                 cap.InitialiseModel,
                 cap.ThetaSynapticStimuli,
                 cap.RunSimulation_ReturnTraces,
                 cap.NumOfPossibleLocations,
                 cap.ReceivesCurrentPulses_ProvidesResponse_MultipleLocations):

    def __init__(self, name="model", mod_files_path=None):
        """ Constructor. """

        """ This class should be used with Jupyter notebooks"""

        self.modelpath = mod_files_path
        self.libpath = "x86_64/.libs/libnrnmech.so.0"
        self.hocpath = None

        self.cvode_active = False

        self.template_name = None
        self.SomaSecList_name = None
        self.max_dist_from_soma = 150
        self.v_init = -70
        self.celsius = 34

        self.name = name
        self.threshold = -20
        self.stim = None
        self.stim_list = [] 
        self.soma = None
        sciunit.Model.__init__(self, name=name)

        self.c_step_start = 0.00004
        self.c_step_stop = 0.000004
        self.c_minmax = numpy.array([0.00004, 0.04])

        self.SecList_name= None
        self.ObliqueSecList_name = None
        self.TrunkSecList_name = None
        self.ApicalSecList_name = None
        self.BasalSecList_name = None
        self.TuftSecList_name = None
        self.dend_loc = []  #self.dend_loc = [['dendrite[80]',0.27],['dendrite[80]',0.83],['dendrite[54]',0.16],['dendrite[54]',0.95],['dendrite[52]',0.38],['dendrite[52]',0.83],['dendrite[53]',0.17],['dendrite[53]',0.7],['dendrite[28]',0.35],['dendrite[28]',0.78]]
        self.dend_locations = collections.OrderedDict()
        self.NMDA_name = None
        self.default_NMDA_name = 'NMDA_CA1_pyr_SC'
        self.default_NMDA_path = pkg_resources.resource_filename("hippounit", "tests/default_NMDAr/")

        self.AMPA_name = None
        self.AMPA_NMDA_ratio = 2.0

        self.AMPA_tau1 = 0.1
        self.AMPA_tau2 = 2.0
        self.start=150

        self.ns = None
        self.ampa = None
        self.nmda = None
        self.ampa_nc = None
        self.nmda_nc = None

        self.ampa_list = []
        self.nmda_list = []
        self.ns_list = []
        self.ampa_nc_list = []
        self.nmda_nc_list = []

        self.synapse_lists = {} 
        self.spine_dict ={}

        self.ndend = None
        self.xloc = None

        self.base_directory = './validation_results/'   # inside current directory

        self.find_section_lists = False

        self.compile_mod_files()
        self.compile_default_NMDA()

    def translate(self, sectiontype, distance=0):

        if "soma" in sectiontype:
            return self.soma
        else:
            return False

    def compile_mod_files(self):

        if self.modelpath is None:
            raise Exception("Please give the path to the mod files (eg. mod_files_path = \"/home/models/CA1_pyr/mechanisms/\") as an argument to the ModelLoader class")

        if os.path.isfile(self.modelpath + self.libpath) is False:
            os.system("cd " + self.modelpath + "; nrnivmodl")

    def compile_default_NMDA(self):
        if os.path.isfile(self.default_NMDA_path + self.libpath) is False:
            os.system("cd " + self.default_NMDA_path + "; nrnivmodl")

    def load_mod_files(self):

        h.nrn_load_dll(str(self.modelpath + self.libpath))


    def initialise(self):

        save_stdout=sys.stdout                   #To supress hoc output from Jupyter notebook 
        # sys.stdout=open("trash","w")
        sys.stdout=open('/dev/stdout', 'w')      #rather print it to the console 

        self.load_mod_files()

        if self.hocpath is None:
            raise Exception("Please give the path to the hoc file (eg. model.modelpath = \"/home/models/CA1_pyr/CA1_pyr_model.hoc\")")


        h.load_file("stdrun.hoc")
        h.load_file(str(self.hocpath))

        if self.soma is None and self.SomaSecList_name is None:
            raise Exception("Please give the name of the soma (eg. model.soma=\"soma[0]\"), or the name of the somatic section list (eg. model.SomaSecList_name=\"somatic\")")

        try:
            if self.template_name is not None and self.SomaSecList_name is not None:

                h('objref testcell')
                h('testcell = new ' + self.template_name)

                exec('self.soma_ = h.testcell.'+ self.SomaSecList_name)

                for s in self.soma_ :
                    self.soma = h.secname()

            elif self.template_name is not None and self.SomaSecList_name is None:
                h('objref testcell')
                h('testcell = new ' + self.template_name)
                # in this case self.soma is set in the jupyter notebook
            elif self.template_name is None and self.SomaSecList_name is not None:
                exec('self.soma_ = h.' +  self.SomaSecList_name)
                for s in self.soma_ :
                    self.soma = h.secname()
            # if both is None, the model is loaded, self.soma will be used
        except AttributeError:
            print ("The provided model template is not accurate. Please verify!")
        except Exception:
            print ("If a model template is used, please give the name of the template to be instantiated (with parameters, if any). Eg. model.template_name=CCell(\"morph_path\")")
            raise


        sys.stdout=save_stdout    #setting output back to normal 

    def inject_current(self, amp, delay, dur, section_stim, loc_stim, section_rec, loc_rec):

        self.initialise()

        if self.cvode_active:
            h.cvode_active(1)
        else:
            h.cvode_active(0)

        stim_section_name = self.translate(section_stim, distance=0)
        rec_section_name = self.translate(section_rec, distance=0)
        #exec("self.sect_loc=h." + str(self.soma)+"("+str(0.5)+")")

        exec("self.sect_loc_stim=h." + str(stim_section_name)+"("+str(loc_stim)+")")

        print("- running amplitude: " + str(amp)  + " on model: " + self.name + " at: " + stim_section_name + "(" + str(loc_stim) + ")")

        self.stim = h.IClamp(self.sect_loc_stim)

        self.stim.amp = amp
        self.stim.delay = delay
        self.stim.dur = dur

        #print "- running model", self.name, "stimulus at: ", str(self.soma), "(", str(0.5), ")"

        exec("self.sect_loc_rec=h." + str(rec_section_name)+"("+str(loc_rec)+")")

        rec_t = h.Vector()
        rec_t.record(h._ref_t)

        rec_v = h.Vector()
        rec_v.record(self.sect_loc_rec._ref_v)

        h.stdinit()

        dt = 0.025
        h.dt = dt
        h.steps_per_ms = 1/dt
        h.v_init = self.v_init#-65

        h.celsius = self.celsius
        h.init()
        h.tstop = delay + dur + 200
        h.run()

        t = numpy.array(rec_t)
        v = numpy.array(rec_v)

        return t, v

    def inject_current_record_respons_multiple_loc(self, amp, delay, dur, section_stim, loc_stim, dend_locations):
       # Modified: An if condition has been added to deal with the different form of dendritic locatinr returned by The          ProvidesRecordingLocationsOnTrunk and the ProvidesRandomDendriticLocations capabilities. 

        self.initialise()

        if self.cvode_active:
            h.cvode_active(1)
        else:
            h.cvode_active(0)

        stim_section_name = self.translate(section_stim, distance=0)
        #rec_section_name = self.translate(section_rec, distance=0)
        #exec("self.sect_loc=h." + str(self.soma)+"("+str(0.5)+")")

        exec("self.sect_loc_stim=h." + str(stim_section_name)+"("+str(loc_stim)+")")
        exec("self.sect_loc_rec=h." + str(stim_section_name)+"("+str(loc_stim)+")")

        print("- running amplitude: " + str(amp)  + " on model: " + self.name + " at: " + stim_section_name + "(" + str(loc_stim) + ")")

        self.stim = h.IClamp(self.sect_loc_stim)

        self.stim.amp = amp
        self.stim.delay = delay
        self.stim.dur = dur

        rec_t = h.Vector()
        rec_t.record(h._ref_t)

        rec_v_stim = h.Vector()
        rec_v_stim.record(self.sect_loc_rec._ref_v)

        rec_v = []
        v = collections.OrderedDict()
        self.dend_loc_rec =[]

        '''
        for i in range(0,len(dend_loc)):

            exec("self.dend_loc_rec.append(h." + str(dend_loc[i][0])+"("+str(dend_loc[i][1])+"))")
            rec_v.append(h.Vector())
            rec_v[i].record(self.dend_loc_rec[i]._ref_v)
            #print self.dend_loc[i]
        '''
        #print dend_locations
        # The ProvidesRecordingLocationsOnTrunk and the ProvidesRandomDendriticLocations capabilities returns the dendritic locations in different format. 

        if isinstance(dend_locations, list):
            for locs in dend_locations:
                exec("self.dend_loc_rec.append(h." + str(locs[0])+"("+str(locs[1])+"))")
                rec_v.append(h.Vector())

        else:
            for key, value in dend_locations.items():
                for i in range(len(dend_locations[key])):
                    exec("self.dend_loc_rec.append(h." + str(dend_locations[key][i][0])+"("+str(dend_locations[key][i][1])+"))")
                    rec_v.append(h.Vector())


        for i in range(len(self.dend_loc_rec)):
            rec_v[i].record(self.dend_loc_rec[i]._ref_v)
            #print self.dend_loc[i]

        h.stdinit()

        dt = 0.025
        h.dt = dt
        h.steps_per_ms = 1/dt
        h.v_init = self.v_init#-65

        h.celsius = self.celsius
        h.init()
        h.tstop = delay + dur + 200
        h.run()

        t = numpy.array(rec_t)
        v_stim = numpy.array(rec_v_stim)

        '''
        for i in range(0,len(dend_loc)):
            v.append(numpy.array(rec_v[i]))
        '''

        # The ProvidesRecordingLocationsOnTrunk and the ProvidesRandomDendriticLocations capabilities returns the dendritic locations in different format. This function's return form depends on which format it gets.

        i = 0
        if isinstance(dend_locations, list):
            for loc in dend_locations:
                loc_key = (loc[0], loc[1]) # list can not be a key, but tuple can
                v[loc_key] = numpy.array(rec_v[i])     # the list that specifies dendritic location will be a key too.
                i+=1
        else:
            for key, value in dend_locations.items():
                v[key] = collections.OrderedDict()
                for j in range(len(dend_locations[key])):
                    loc_key = (dend_locations[key][j][0],dend_locations[key][j][1]) # list can not be a key, but tuple can
                    v[key][loc_key] = numpy.array(rec_v[i])     # the list that specifies dendritic location will be a key too.
                    i+=1

        return t, v_stim, v

    def inject_current_pulses_record_respons_multiple_loc(self, amp, delay, dur_of_pulse, dur_of_stim, num_of_pulses, frequency, section_stim, loc_stim, dend_locations): 

        self.initialise()

        if self.cvode_active:
            h.cvode_active(1)
        else:
            h.cvode_active(0)

        stim_section_name = self.translate(section_stim, distance=0)
        #rec_section_name = self.translate(section_rec, distance=0)
        #exec("self.sect_loc=h." + str(self.soma)+"("+str(0.5)+")")

        exec("self.sect_loc_stim=h." + str(stim_section_name)+"("+str(loc_stim)+")")
        exec("self.sect_loc_rec=h." + str(stim_section_name)+"("+str(loc_stim)+")")

        print("- running amplitude: " + str(amp) + " with " + str(frequency) + " Hz"  + " on model: " + self.name + " at: " + stim_section_name + "(" + str(loc_stim) + ")")

        interval_bw_pulses = 1.0/ frequency * 1000.0
        self.stim_list = [None] * num_of_pulses

        for i in range(num_of_pulses):
            self.stim_list[i] = h.IClamp(self.sect_loc_stim)

            self.stim_list[i].amp = amp
            self.stim_list[i].delay = delay + i * interval_bw_pulses
            self.stim_list[i].dur = dur_of_pulse

        rec_t = h.Vector()
        rec_t.record(h._ref_t)

        rec_v_stim = h.Vector()
        rec_v_stim.record(self.sect_loc_rec._ref_v)

        rec_v = []
        v = collections.OrderedDict()
        self.dend_loc_rec =[]

        '''
        for i in range(0,len(dend_loc)):

            exec("self.dend_loc_rec.append(h." + str(dend_loc[i][0])+"("+str(dend_loc[i][1])+"))")
            rec_v.append(h.Vector())
            rec_v[i].record(self.dend_loc_rec[i]._ref_v)
            #print self.dend_loc[i]
        '''
        #print dend_locations
        # The ProvidesRecordingLocationsOnTrunk and the ProvidesRandomDendriticLocations capabilities returns the dendritic locations in different format. 

        if isinstance(dend_locations, list):
            for locs in dend_locations:
                exec("self.dend_loc_rec.append(h." + str(locs[0])+"("+str(locs[1])+"))")
                rec_v.append(h.Vector())

        else:
            for key, value in dend_locations.items():
                for i in range(len(dend_locations[key])):
                    exec("self.dend_loc_rec.append(h." + str(dend_locations[key][i][0])+"("+str(dend_locations[key][i][1])+"))")
                    rec_v.append(h.Vector())


        for i in range(len(self.dend_loc_rec)):
            rec_v[i].record(self.dend_loc_rec[i]._ref_v)
            #print self.dend_loc[i]

        h.stdinit()

        dt = 0.025
        h.dt = dt
        h.steps_per_ms = 1/dt
        h.v_init = self.v_init#-65

        h.celsius = self.celsius
        h.init()
        h.tstop = delay + dur_of_stim + 200
        h.run()

        t = numpy.array(rec_t)
        v_stim = numpy.array(rec_v_stim)

        '''
        for i in range(0,len(dend_loc)):
            v.append(numpy.array(rec_v[i]))
        '''

        # The ProvidesRecordingLocationsOnTrunk and the ProvidesRandomDendriticLocations capabilities returns the dendritic locations in different format. This function's return form depends on which format it gets.

        i = 0
        if isinstance(dend_locations, list):
            for loc in dend_locations:
                loc_key = (loc[0], loc[1]) # list can not be a key, but tuple can
                v[loc_key] = numpy.array(rec_v[i])     # the list that specifies dendritic location will be a key too.
                i+=1
        else:
            for key, value in dend_locations.items():
                v[key] = collections.OrderedDict()
                for j in range(len(dend_locations[key])):
                    loc_key = (dend_locations[key][j][0],dend_locations[key][j][1]) # list can not be a key, but tuple can
                    v[key][loc_key] = numpy.array(rec_v[i])     # the list that specifies dendritic location will be a key too.
                    i+=1

        return t, v_stim, v

    def classify_apical_point_sections(self, icell):

        import os
        import neurom as nm
        from hippounit import classify_apical_sections as cas

        '''
        for file_name in os.listdir(self.morph_path[1:-1]):
            filename = self.morph_path[1:-1]+ '/' + file_name
            break
        '''

        morph = nm.load_neuron(self.morph_full_path)

        apical_point_sections = cas.multiple_apical_points(morph)

        sections = cas.get_list_of_diff_section_types(morph, apical_point_sections)

        apical_trunk_isections = cas.get_neuron_isections(icell, sections['trunk'])
        #print sorted(apical_trunk_isections)

        apical_tuft_isections = cas.get_neuron_isections(icell, sections['tuft'])
        #print sorted(apical_tuft_isections)

        oblique_isections = cas.get_neuron_isections(icell, sections['obliques'])
        #print sorted(oblique_isections)

        return apical_trunk_isections, apical_tuft_isections, oblique_isections

    def find_trunk_locations(self, distances, tolerance):

        if self.TrunkSecList_name is None and not self.find_section_lists:
            raise NotImplementedError("Please give the name of the section list containing the trunk sections. (eg. model.TrunkSecList_name=\"trunk\" or set model.find_section_lists to True)")

        #locations={}
        locations=collections.OrderedDict()
        actual_distances ={}
        dend_loc=[]

        if self.TrunkSecList_name is not None:
            self.initialise()

            if self.template_name is not None:
                exec('self.trunk=h.testcell.' + self.TrunkSecList_name)
            else:
                exec('self.trunk=h.' + self.TrunkSecList_name)


        if self.find_section_lists:

            self.initialise()

            if self.template_name is not None:
                exec('self.icell=h.testcell')

            apical_trunk_isections, apical_tuft_isections, oblique_isections = self.classify_apical_point_sections(self.icell)

            self.trunk = []
            for i in range(len(apical_trunk_isections)):
                exec('self.sec = h.testcell.apic[' + str(apical_trunk_isections[i]) + ']')
                self.trunk.append(self.sec)

        for sec in self.trunk:
            #for seg in sec:
            h(self.soma + ' ' +'distance()') #set soma as the origin
            #print sec.name()
            if self.find_section_lists:
                h('access ' + sec.name())

            for seg in sec:
                #print 'SEC: ', sec.name(),
                #print 'SEG.X', seg.x
                #print 'DIST', h.distance(seg.x)
                #print 'DIST0', h.distance(0)
                #print 'DIST1', h.distance(1)
                for i in range(0, len(distances)):
                    locations.setdefault(distances[i], []) # if this key doesn't exist it is added with the value: [], if exists, value not altered
                    if h.distance(seg.x) < (distances[i] + tolerance) and h.distance(seg.x) > (distances[i]- tolerance): # if the seq is between distance +- 20
                        #print 'SEC: ', sec.name()
                        #print 'seg.x: ', seg.x
                        #print 'DIST: ', h.distance(seg.x)
                        locations[distances[i]].append([sec.name(), seg.x])
                        actual_distances[sec.name(), seg.x] = h.distance(seg.x)

        #print actual_distances
        return locations, actual_distances

    def get_random_locations(self, dendritic_type, num, seed, dist_range):

        # modified to make it more general and useable for any of the main dendritic types
       # TODO: modify original tests to adapt to this, and check if everything works fine (AND PathwayInteractionTest)

        if dendritic_type == 'basal':
            SecList_name = self.BasalSecList_name

            if SecList_name is None:    # find_section_list can not be used for basal dendrites, as it classifies apical dendrites into subtypes
                raise NotImplementedError("Please give the name of the section list containing the basal dendritic sections. (eg. model.BasalSecList_name=\"basal\" or set model.find_section_lists to True)")

        elif dendritic_type == 'apical':
            SecList_name = self.ApicalSecList_name

            if SecList_name is None: # find_section_list can not be used for basal dendrites, as it classifies apical dendrites into subtypes
                raise NotImplementedError("Please give the name of the section list containing the apical dendritic sections. (eg. model.ApicalSecList_name=\"apical\" or set model.find_section_lists to True)")

        elif dendritic_type == 'trunk':
            SecList_name = self.TrunkSecList_name

            if SecList_name is None and not self.find_section_lists:
                raise NotImplementedError("Please give the name of the section list containing the trunk sections. (eg. model.TrunkSecList_name=\"trunk\" or set model.find_section_lists to True)")

        elif dendritic_type == 'tuft':
            SecList_name = self.TuftSecList_name

            if SecList_name is None and not self.find_section_lists:
                raise NotImplementedError("Please give the name of the section list containing the tuft sections. (eg. model.TuftkSecList_name=\"tuft\" or set model.find_section_lists to True)")

        elif dendritic_type == 'oblique':
            SecList_name = self.ObliqueSecList_name

            if SecList_name is None and not self.find_section_lists:
                raise NotImplementedError("Please give the name of the section list containing the oblique dendritic sections. (eg. model.ObliqueSecList_name=\"oblique\" or set model.find_section_lists to True)")


        locations=[]
        locations_distances = collections.OrderedDict()

        if SecList_name is not None:
            self.initialise()

            if self.template_name is not None:
                exec('self.dendrites=h.testcell.' + SecList_name)

            else:
                exec('self.dendrites=h.' + SecList_name)

        if self.find_section_lists and (dendritic_type == 'tuft' or dendritic_type == 'trunk' or dendritic_type == 'oblique'): # The classification is only for apical dendrites, (not basal) 

            self.initialise()

            if self.template_name is not None:
                exec('self.icell=h.testcell')

            apical_trunk_isections, apical_tuft_isections, oblique_isections = self.classify_apical_point_sections(self.icell)
            apical_trunk_isections = sorted(apical_trunk_isections) # important to keep reproducability

            self.dendrites = []
            if dendritic_type == 'trunk':
                for i in range(len(apical_trunk_isections)):
                    exec('self.sec = h.testcell.apic[' + str(apical_trunk_isections[i]) + ']')
                    self.dendrites.append(self.sec)
            elif dendritic_type == 'tuft':
                for i in range(len(apical_tuft_isections)):
                    exec('self.sec = h.testcell.apic[' + str(apical_tuft_isections[i]) + ']')
                    self.dendrites.append(self.sec)
            elif dendritic_type == 'oblique':
                for i in range(len(oblique_isections)):
                    exec('self.sec = h.testcell.apic[' + str(oblique_isections[i]) + ']')
                    self.dendrites.append(self.sec)
        else:
            self.dendrites = list(self.dendrites)

        kumm_length_list = []
        kumm_length = 0
        num_of_secs = 0


        for sec in self.dendrites:
            #print sec.L
            num_of_secs += sec.nseg
            kumm_length += sec.L
            kumm_length_list.append(kumm_length)
        #print 'kumm' ,kumm_length_list
        #print num_of_secs

        if num > num_of_secs:
            for sec in self.dendrites:
                h(self.soma + ' ' +'distance()')
                h('access ' + sec.name())
                for seg in sec:
                    if h.distance(seg.x) > dist_range[0] and h.distance(seg.x) < dist_range[1]:     # if they are out of the distance range they wont be used
                        locations.append([sec.name(), seg.x])
                        locations_distances[sec.name(), seg.x] = h.distance(seg.x)
            #print 'Dendritic locations to be tested (with their actual distances):', locations_distances

        else:

            norm_kumm_length_list = [i/kumm_length_list[-1] for i in kumm_length_list]
            #print 'norm kumm',  norm_kumm_length_list

            import random

            _num_ = num  # _num_ will be changed
            num_iterations = 0

            while len(locations) < num and num_iterations < 50 :
                #print 'seed ', seed
                random.seed(seed)
                rand_list = [random.random() for j in range(_num_)]
                #print rand_list

                for rand in rand_list:
                    #print 'RAND', rand
                    for i in range(len(norm_kumm_length_list)):
                        if rand <= norm_kumm_length_list[i] and (rand > norm_kumm_length_list[i-1] or i==0):
                            #print norm_kumm_length_list[i-1]
                            #print norm_kumm_length_list[i]
                            seg_loc = (rand - norm_kumm_length_list[i-1]) / (norm_kumm_length_list[i] - norm_kumm_length_list[i-1])
                            #print 'seg_loc', seg_loc
                            segs = [seg.x for seg in self.dendrites[i]]
                            d_seg = [abs(seg.x - seg_loc) for seg in self.dendrites[i]]
                            min_d_seg = numpy.argmin(d_seg)
                            segment = segs[min_d_seg]
                            #print 'segment', segment
                            h(self.soma + ' ' +'distance()')
                            h('access ' + self.dendrites[i].name())
                            if [self.dendrites[i].name(), segment] not in locations and h.distance(segment) >= dist_range[0] and h.distance(segment) < dist_range[1]:
                                locations.append([self.dendrites[i].name(), segment])
                                locations_distances[self.dendrites[i].name(), segment] = h.distance(segment)
                _num_ = num - len(locations)
                #print '_num_', _num_
                seed += 10
                num_iterations += 1
                #print len(locations)
        #print 'Dendritic locations to be tested (with their actual distances):', locations_distances

        return locations, locations_distances

    def find_good_obliques(self):
        """Used in ObliqueIntegrationTest"""

        if (self.ObliqueSecList_name is None or self.TrunkSecList_name is None) and not self.find_section_lists:
            raise NotImplementedError("Please give the names of the section lists containing the oblique dendrites and the trunk sections. (eg. model.ObliqueSecList_name=\"obliques\", model.TrunkSecList_name=\"trunk\" or set model.find_section_lists to True)")


        #self.initialise()

        good_obliques = h.SectionList()
        dend_loc=[]

        if self.TrunkSecList_name is not None and self.ObliqueSecList_name is not None:
            self.initialise()

            if self.template_name is not None:

                exec('self.oblique_dendrites=h.testcell.' + self.ObliqueSecList_name)   # so we can have the name of the section list as a string given by the user
                #exec('oblique_dendrites = h.' + oblique_seclist_name)
                exec('self.trunk=h.testcell.' + self.TrunkSecList_name)
            else:
                exec('self.oblique_dendrites=h.' + self.ObliqueSecList_name)   # so we can have the name of the section list as a string given by the user
                #exec('oblique_dendrites = h.' + oblique_seclist_name)
                exec('self.trunk=h.' + self.TrunkSecList_name)

        if self.find_section_lists:

            self.initialise()

            if self.template_name is not None:
                exec('self.icell=h.testcell')

            apical_trunk_isections, apical_tuft_isections, oblique_isections = self.classify_apical_point_sections(self.icell)

            self.trunk = []
            for i in range(len(apical_trunk_isections)):
                exec('self.sec = h.testcell.apic[' + str(apical_trunk_isections[i]) + ']')
                self.trunk.append(self.sec)

            self.oblique_dendrites = []
            for i in range(len(oblique_isections)):
                exec('self.sec = h.testcell.apic[' + str(oblique_isections[i]) + ']')
                self.oblique_dendrites.append(self.sec)

        good_obliques_added = 0

        while good_obliques_added == 0 and self.max_dist_from_soma <= 190:
            for sec in self.oblique_dendrites:
                h(self.soma + ' ' +'distance()') #set soma as the origin
                if self.find_section_lists:
                    h('access ' + sec.name())
                parent = h.SectionRef(sec).parent
                child_num = h.SectionRef(sec).nchild()
                dist = h.distance(0)
                #print 'SEC: ', sec.name()
                #print 'NCHILD: ', child_num
                #print 'PARENT: ', parent.name()
                #print 'DIST: ', h.distance(0)
                """
                for trunk_sec in trunk:
                    if self.find_section_lists:
                        h('access ' + trunk_sec.name())
                    if h.issection(parent.name()) and dist < self.max_dist_from_soma and child_num == 0:   # true if string (parent.name()) is contained in the name of the currently accessed section.trunk_sec is the accessed section,
                        #print sec.name(), parent.name()
                        h('access ' + sec.name())         # only currently accessed section can be added to hoc SectionList
                        good_obliques.append(sec.name())
                        good_obliques_added += 1
                """
                if dist < self.max_dist_from_soma and child_num == 0:   # now the oblique section can branch from another oblique section, but it has to be a tip (terminal) section
                    #print sec.name(), parent.name()
                    # print sec.name(), dist
                    h('access ' + sec.name())         # only currently accessed section can be added to hoc SectionList
                    good_obliques.append(sec.name())
                    good_obliques_added += 1
            if good_obliques_added == 0:
                self.max_dist_from_soma += 15
                print("Maximum distance from soma was increased by 15 um, new value: " + str(self.max_dist_from_soma))

        for sec in good_obliques:

            dend_loc_prox=[]
            dend_loc_dist=[]
            seg_list_prox=[]
            seg_list_dist=[]

            h(sec.name() + ' ' +'distance()')  #set the 0 point of the section as the origin
            # print(sec.name())


            for seg in sec:
                # print(seg.x, h.distance(seg.x))
                if h.distance(seg.x) > 5 and h.distance(seg.x) < 50:
                    seg_list_prox.append(seg.x)
                if h.distance(seg.x) > 60 and h.distance(seg.x) < 126:
                    seg_list_dist.append(seg.x)

            #print seg_list_prox
            #print seg_list_dist

            if len(seg_list_prox) > 1:
                s = int(numpy.ceil(len(seg_list_prox)/2.0))
                dend_loc_prox.append(sec.name())
                dend_loc_prox.append(seg_list_prox[s])
                dend_loc_prox.append('prox')
            elif len(seg_list_prox) == 1:
                dend_loc_prox.append(sec.name())
                dend_loc_prox.append(seg_list_prox[0])
                dend_loc_prox.append('prox')

            if len(seg_list_dist) > 1:
                s = int(numpy.ceil(len(seg_list_dist)/2.0)-1)
                dend_loc_dist.append(sec.name())
                dend_loc_dist.append(seg_list_dist[s])
                dend_loc_dist.append('dist')
            elif len(seg_list_dist) == 1:
                dend_loc_dist.append(sec.name())
                dend_loc_dist.append(seg_list_dist[0])
                dend_loc_dist.append('dist')
            elif len(seg_list_dist) == 0:                # if the dendrite is not long enough to meet the criteria, we stimulate its end
                dend_loc_dist.append(sec.name())
                dend_loc_dist.append(0.9)
                dend_loc_dist.append('dist')

            if dend_loc_prox:
                dend_loc.append(dend_loc_prox)
            if dend_loc_dist:
                dend_loc.append(dend_loc_dist)

        #print 'Dendrites and locations to be tested: ', dend_loc

        return dend_loc


    def set_ampa_nmda(self, dend_loc):
        """Currently not used - Used to be used in ObliqueIntegrationTest"""

        ndend, xloc, loc_type = dend_loc

        exec("self.dendrite=h." + ndend)

        self.ampa = h.Exp2Syn(xloc, sec=self.dendrite)
        self.ampa.tau1 = self.AMPA_tau1
        self.ampa.tau2 = self.AMPA_tau2

        exec("self.nmda = h."+self.NMDA_name+"(xloc, sec=self.dendrite)")

        self.ndend = ndend
        self.xloc = xloc


    def set_netstim_netcon(self, interval):
        """Currently not used - Used to be used in ObliqueIntegrationTest"""

        self.ns = h.NetStim()
        self.ns.interval = interval
        self.ns.number = 0
        self.ns.start = self.start

        self.ampa_nc = h.NetCon(self.ns, self.ampa, 0, 0, 0)
        self.nmda_nc = h.NetCon(self.ns, self.nmda, 0, 0, 0)


    def set_num_weight(self, number, AMPA_weight):
        """Currently not used - Used to be used in ObliqueIntegrationTest"""

        self.ns.number = number
        self.ampa_nc.weight[0] = AMPA_weight
        self.nmda_nc.weight[0] =AMPA_weight/self.AMPA_NMDA_ratio

    def run_syn(self, dend_loc, interval, number, AMPA_weight):
        """Currently not used - Used to be used in ObliqueIntegrationTest"""

        self.initialise()

        if self.cvode_active:
            h.cvode_active(1)
        else:
            h.cvode_active(0)

        self.set_ampa_nmda(dend_loc)
        self.set_netstim_netcon(interval)
        self.set_num_weight(number, AMPA_weight)

        exec("self.sect_loc=h." + str(self.soma)+"("+str(0.5)+")")

        # initiate recording
        rec_t = h.Vector()
        rec_t.record(h._ref_t)

        rec_v = h.Vector()
        rec_v.record(self.sect_loc._ref_v)

        rec_v_dend = h.Vector()
        rec_v_dend.record(self.dendrite(self.xloc)._ref_v)

        h.stdinit()

        dt = 0.025
        h.dt = dt
        h.steps_per_ms = 1/ dt
        h.v_init = self.v_init #-80

        h.celsius = self.celsius
        h.init()
        h.tstop = 500
        h.run()

        # get recordings
        t = numpy.array(rec_t)
        v = numpy.array(rec_v)
        v_dend = numpy.array(rec_v_dend)

        return t, v, v_dend

    def set_multiple_ampa_nmda(self, dend_loc, number):
        """Used in ObliqueIntegrationTest"""

        ndend, xloc, loc_type = dend_loc

        exec("self.dendrite=h." + ndend)

        for i in range(number):

            if self.AMPA_name: # if this is given, the AMPA model defined in a mod file is used, else the built in Exp2Syn
                exec("self.ampa_list[i] = h."+self.AMPA_name+"(xloc, sec=self.dendrite)")
            else:
                self.ampa_list[i] = h.Exp2Syn(xloc, sec=self.dendrite)
                self.ampa_list[i].tau1 = self.AMPA_tau1
                self.ampa_list[i].tau2 = self.AMPA_tau2
                #print 'The built in Exp2Syn is used as the AMPA component. Tau1 = ', self.AMPA_tau1, ', Tau2 = ', self.AMPA_tau2 , '.'

            if self.NMDA_name: # if this is given, the NMDA model defined in a mod file is used, else the default NMDA model of HippoUnit
                exec("self.nmda_list[i] = h."+self.NMDA_name+"(xloc, sec=self.dendrite)")
            else:
                try:
                    exec("self.nmda_list[i] = h."+self.default_NMDA_name+"(xloc, sec=self.dendrite)")
                except:
                    h.nrn_load_dll(self.default_NMDA_path + self.libpath)
                    exec("self.nmda_list[i] = h."+self.default_NMDA_name+"(xloc, sec=self.dendrite)")

        self.ndend = ndend
        self.xloc = xloc


    def set_multiple_netstim_netcon(self, interval, number, AMPA_weight):
        """Used in ObliqueIntegrationTest"""

        for i in range(number):
            self.ns_list[i] = h.NetStim()
            self.ns_list[i].number = 1
            self.ns_list[i].start = self.start + (i*interval)

            self.ampa_nc_list[i] = h.NetCon(self.ns_list[i], self.ampa_list[i], 0, 0, 0)
            self.nmda_nc_list[i] = h.NetCon(self.ns_list[i], self.nmda_list[i], 0, 0, 0)

            self.ampa_nc_list[i].weight[0] = AMPA_weight
            self.nmda_nc_list[i].weight[0] =AMPA_weight/self.AMPA_NMDA_ratio


    def run_multiple_syn(self, dend_loc, interval, number, weight):
        """Used in ObliqueIntegrationTest"""

        self.ampa_list = [None] * number
        self.nmda_list = [None] * number
        self.ns_list = [None] * number
        self.ampa_nc_list = [None] * number
        self.nmda_nc_list = [None] * number


        self.initialise()

        if self.cvode_active:
            h.cvode_active(1)
        else:
            h.cvode_active(0)

        self.set_multiple_ampa_nmda(dend_loc, number)

        self.set_multiple_netstim_netcon(interval, number, weight)


        exec("self.sect_loc=h." + str(self.soma)+"("+str(0.5)+")")

        # initiate recording
        rec_t = h.Vector()
        rec_t.record(h._ref_t)

        rec_v = h.Vector()
        rec_v.record(self.sect_loc._ref_v)

        rec_v_dend = h.Vector()
        rec_v_dend.record(self.dendrite(self.xloc)._ref_v)

        h.stdinit()

        dt = 0.025
        h.dt = dt
        h.steps_per_ms = 1/dt
        h.v_init = self.v_init #-80

        h.celsius = self.celsius
        h.init()
        h.tstop =500
        h.run()

        # get recordings
        t = numpy.array(rec_t)
        v = numpy.array(rec_v)
        v_dend = numpy.array(rec_v_dend)

        return t, v, v_dend



    def set_Exp2Syn(self, dend_loc, tau1, tau2):
        """Used in PSPAttenuationTest"""

        ndend, xloc = dend_loc

        exec("self.dendrite=h." + ndend)

        self.ampa = h.Exp2Syn(xloc, sec=self.dendrite)
        self.ampa.tau1 = tau1
        self.ampa.tau2 = tau2

        self.ndend = ndend
        self.xloc = xloc


    def set_netstim_netcon_Exp2Syn(self):
        """Used in PSPAttenuationTest"""
        self.start = 300

        self.ns = h.NetStim()
        #self.ns.interval = interval
        #self.ns.number = 0
        self.ns.start = self.start

        self.ampa_nc = h.NetCon(self.ns, self.ampa, 0, 0, 0)

    def set_weight_Exp2Syn(self, weight):
        """Used in PSPAttenuationTest"""

        self.ns.number = 1
        self.ampa_nc.weight[0] = weight

    def run_EPSCstim(self, dend_loc, weight, tau1, tau2):
        """Used in PSPAttenuationTest"""

        self.initialise()

        if self.cvode_active:
            h.cvode_active(1)
        else:
            h.cvode_active(0)

        self.set_Exp2Syn(dend_loc, tau1, tau2)
        self.set_netstim_netcon_Exp2Syn()
        self.set_weight_Exp2Syn(weight)

        exec("self.sect_loc=h." + str(self.soma)+"("+str(0.5)+")")

        # initiate recording
        rec_t = h.Vector()
        rec_t.record(h._ref_t)

        rec_v = h.Vector()
        rec_v.record(self.sect_loc._ref_v)

        rec_v_dend = h.Vector()
        rec_v_dend.record(self.dendrite(self.xloc)._ref_v)

        h.stdinit()

        dt = 0.025
        h.dt = dt
        h.steps_per_ms = 1/dt
        h.v_init = self.v_init #-80

        h.celsius = self.celsius
        h.init()
        h.tstop = 450
        h.run()

        # get recordings
        t = numpy.array(rec_t)
        v = numpy.array(rec_v)
        v_dend = numpy.array(rec_v_dend)

        return t, v, v_dend

class ModelLoader_BPO(ModelLoader):

    def __init__(self, name="model", model_dir=None, SomaSecList_name=None):
        """ Constructor. """
        """ This class should be used with Jupyter notebooks"""
        super(ModelLoader_BPO, self).__init__(name=name)
        self.SomaSecList_name = SomaSecList_name
        self.morph_full_path = None
        self.find_section_lists = True

        self.setup_dirs(model_dir)
        self.setup_values()
        self.compile_mod_files_BPO()

    def compile_mod_files(self):
        """This method is called by the parent class (ModelLoader), but as the path to the mod files is unknown at this point, this is not used. compile_mode_files_BPO is used instead to compile the mod files."""
        pass

    def compile_mod_files_BPO(self):

        if self.modelpath is None:
            raise Exception("Please give the path to the mod files (eg. model.modelpath = \"/home/models/CA1_pyr/mechanisms/\")")

        if os.path.isfile(self.modelpath + self.libpath) is False:
            os.system("cd " + self.modelpath + "; nrnivmodl")

    def load_mod_files(self):

        h.nrn_load_dll(str(self.modelpath + self.libpath))

    def setup_dirs(self, model_dir=""):

        '''
        split_dir = model_dir.split('/')
        del split_dir[-1]
        outer_dir = '/'.join(split_dir)

        if not os.path.exists(model_dir):
            try:

                #split_dir = model_dir.split('/')
                #del split_dir[-1]
                #outer_dir = '/'.join(split_dir)

                zip_ref = zipfile.ZipFile(model_dir + '.zip', 'r')
                zip_ref.extractall(outer_dir)
            except IOError:
                print "Error accessing directory/zipfile named: ", model_dir
        '''

        base_path = os.path.join(model_dir, self.name)
        if os.path.exists(base_path) or os.path.exists(base_path+".zip"):     # If the model_dir is the outer directory, that contains the zip
            self.base_path = base_path
            if not os.path.exists(self.base_path):
                file_ref = zipfile.ZipFile(self.base_path+".zip", 'r')
                file_ref.extractall(model_dir)
                file_ref.close()
            try:
                with open(self.base_path + '/' + self.name + '_meta.json') as f:
                    meta_data = json.load(f, object_pairs_hook=collections.OrderedDict)
            except Exception as e1:
                try:
                    with open(model_dir + '/' + self.name + '_meta.json') as f:
                        meta_data = json.load(f, object_pairs_hook=collections.OrderedDict)
                except Exception as e2:
                    print(e1,e2)
        else:                                                                   # If model_dir is the inner directory (already unzipped)
            self.base_path = model_dir
            split_dir = model_dir.split('/')
            del split_dir[-1]
            outer_dir = '/'.join(split_dir)

            try:
                with open(self.base_path + '/' + self.name + '_meta.json') as f:
                    meta_data = json.load(f, object_pairs_hook=collections.OrderedDict)
            except Exception as e1:
                try:
                    with open(outer_dir + '/' + self.name + '_meta.json') as f:
                        meta_data = json.load(f, object_pairs_hook=collections.OrderedDict)
                except Exception as e2:
                    print(e1,e2)
        '''
        try:
            with open(self.base_path + '/' + self.name + '_meta.json') as f:
                meta_data = json.load(f, object_pairs_hook=collections.OrderedDict)
        except:
            try:
                with open(model_dir + '/' + self.name + '_meta.json') as f:
                    meta_data = json.load(f, object_pairs_hook=collections.OrderedDict)
            except Exception as e:
                print e
        '''

        self.morph_path = "\"" + self.base_path + "/morphology\""

        for file_name in os.listdir(self.morph_path[1:-1]):
            self.morph_full_path = self.morph_path[1:-1]+ '/' + file_name
            break


        # path to mod files
        self.modelpath = self.base_path + "/mechanisms/"

        # if this doesn't exist mod files are automatically compiled
        self.libpath = "x86_64/.libs/libnrnmech.so.0"

        best_cell = meta_data["best_cell"]

        self.hocpath = self.base_path + "/checkpoints/" + str(best_cell)

        if not os.path.exists(self.hocpath):
            self.hocpath = None
            for file in os.listdir(self.base_path + "/checkpoints/"):
                if file.startswith("cell") and file.endswith(".hoc"):
                    self.hocpath = self.base_path + "/checkpoints/" + file
                    print("Model = " + self.name + ": cell.hoc not found in /checkpoints; using " + file)
                    break
            if not os.path.exists(self.hocpath):
                raise IOError("No appropriate .hoc file found in /checkpoints")

        self.base_directory = self.base_path +'/validation_results/'

    def setup_values(self):

        # get model template name
        # could also do this via other JSON, but morph.json seems dedicated for template info
        with open(os.path.join(self.base_path, "config", "morph.json")) as morph_file:
            template_name = list(json.load(morph_file, object_pairs_hook=collections.OrderedDict).keys())[0]

        self.template_name = template_name + "(" + self.morph_path+")"

        # access model config info
        with open(os.path.join(self.base_path, "config", "parameters.json")) as params_file:
            params_data = json.load(params_file, object_pairs_hook=collections.OrderedDict)

        # extract v_init and celsius (if available)
        v_init = None
        celsius = None
        try:
            for item in params_data[template_name]["fixed"]["global"]:
                # would have been better if info was stored inside a dict (rather than a list)
                if "v_init" in item:
                    item.remove("v_init")
                    v_init = float(item[0])
                if "celsius" in item:
                    item.remove("celsius")
                    celsius = float(item[0])
        except:
            pass
        if v_init == None:
            self.v_init = -70.0
            print("Could not find model specific info for `v_init`; using default value of {} mV".format(str(self.v_init)))
        else:
            self.v_init = v_init
        if celsius == None:
            self.celsius = 34.0
            print("Could not find model specific info for `celsius`; using default value of {} degrees Celsius".format(str(self.celsius)))
        else:
            self.celsius = celsius



class ModelLoader_Spine_syn(ModelLoader):

    def __init__(self, name="model", mod_files_path=None):
        """ Constructor. """
        """ This class should be used with Jupyter notebooks"""
        super(ModelLoader_Spine_syn, self).__init__(name=name, mod_files_path=mod_files_path)


        self.start=400

        self.SecList = None

    """ Inputs are on different spines"""
 
    '''
    def block_Na(self):
        h("forsec all_dendrites {gmax_Na_BG_dend = 0.0}")
        h("soma {gmax_Na_BG_soma = 0.0}")
        h("forsec all_axon{gmax_Na_BG_axon = 0.0}")
    '''


    def get_random_locations(self, num, seed, dist_range):

        if self.SecList is None:
            raise NotImplementedError("Please give the name of the section list containing the dendritic sections of interest. (eg. model.SecList=\"trunk\"")

        locations=[]
        locations_distances = {}

        if self.SecList is not None:
            self.initialise()

            if self.template_name is not None:
                exec('self.dendrites=h.testcell.' + self.SecList)

            else:
                exec('self.dendrites=h.' + self.SecList)
        '''
        if self.find_section_lists:

            self.initialise()

            if self.template_name is not None:
                exec('self.icell=h.testcell')

            apical_trunk_isections, apical_tuft_isections, oblique_isections = self.classify_apical_point_sections(self.icell)
            apical_trunk_isections = sorted(apical_trunk_isections) # important to keep reproducability

            self.trunk = []
            for i in range(len(apical_trunk_isections)):
                exec('self.sec = h.testcell.apic[' + str(apical_trunk_isections[i]) + ']')
                self.trunk.append(self.sec)
        else:
        
            self.dendrites = list(self.dendrites)
        '''
        self.dendrites = list(self.dendrites)

        kumm_length_list = []
        kumm_length = 0
        num_of_secs = 0


        for sec in self.dendrites:
            #print sec.L
            num_of_secs += sec.nseg
            kumm_length += sec.L
            kumm_length_list.append(kumm_length)
        #print 'kumm' ,kumm_length_list
        #print num_of_secs

        if num > num_of_secs:
            for sec in self.dendrites:
                h(self.soma + ' ' +'distance()')
                h('access ' + sec.name())
                for seg in sec:
                    if h.distance(seg.x) > dist_range[0] and h.distance(seg.x) < dist_range[1]:     # if they are out of the distance range they wont be used
                        locations.append([sec.name(), seg.x])
                        locations_distances[sec.name(), seg.x] = h.distance(seg.x)
            #print 'Dendritic locations to be tested (with their actual distances):', locations_distances

        else:

            norm_kumm_length_list = [i/kumm_length_list[-1] for i in kumm_length_list]
            #print 'norm kumm',  norm_kumm_length_list

            import random

            _num_ = num  # _num_ will be changed
            num_iterations = 0

            while len(locations) < num and num_iterations < 50 :
                #print 'seed ', seed
                random.seed(seed)
                rand_list = [random.random() for j in range(_num_)]
                #print rand_list

                for rand in rand_list:
                    #print 'RAND', rand
                    for i in range(len(norm_kumm_length_list)):
                        if rand <= norm_kumm_length_list[i] and (rand > norm_kumm_length_list[i-1] or i==0):
                            #print norm_kumm_length_list[i-1]
                            #print norm_kumm_length_list[i]
                            seg_loc = (rand - norm_kumm_length_list[i-1]) / (norm_kumm_length_list[i] - norm_kumm_length_list[i-1])
                            #print 'seg_loc', seg_loc
                            segs = [seg.x for seg in self.dendrites[i]]
                            d_seg = [abs(seg.x - seg_loc) for seg in self.dendrites[i]]
                            min_d_seg = numpy.argmin(d_seg)
                            segment = segs[min_d_seg]
                            #print 'segment', segment
                            h(self.soma + ' ' +'distance()')
                            h('access ' + self.dendrites[i].name())
                            if [self.dendrites[i].name(), segment] not in locations and h.distance(segment) >= dist_range[0] and h.distance(segment) < dist_range[1]:
                                locations.append([self.dendrites[i].name(), segment])
                                locations_distances[self.dendrites[i].name(), segment] = h.distance(segment)
                _num_ = num - len(locations)
                #print '_num_', _num_
                seed += 10
                num_iterations += 1
                #print len(locations)
        #print 'Dendritic locations to be tested (with their actual distances):', locations_distances

        return locations, locations_distances

    def create_single_spine(self, dend_loc):

        ndend, xloc = dend_loc

        exec("self.dendrite=h." + ndend)

        # dist_bw_spines = 0.284 # um
        # length = self.dendrite.L
        # relative_dist_bw_spines = dist_bw_spines / length
        # print "length", length
        # print 'relative dist of spines: ', relative_dist_bw_spines


        # self.sneck = [h.Section(name='sneck[%d]' % i) for i in xrange(max_num_syn)]
        # self.shead = [h.Section(name='shead[%d]' % i) for i in xrange(max_num_syn)]

        self.sneck = h.Section(name='sneck')
        self.shead = h.Section(name='shead')

        #print "e_Leak_pyr", self.dendrite.e_Leak_pyr
        #print "gmax_Leak_pyr", self.dendrite.gmax_Leak_pyr
        #print "cm", self.dendrite.cm


        self.sneck.L = h.sneck_len # 1.58
        self.sneck.diam = h.sneck_diam # 0.077
        self.shead.L = h.shead_len #0.5
        self.shead.diam = h.shead_diam #0.5

        self.sneck.insert('Leak_pyr')
        self.sneck.e_Leak_pyr = self.dendrite.e_Leak_pyr
        self.sneck.gmax_Leak_pyr = self.dendrite.gmax_Leak_pyr
        self.sneck.Ra = 100
        self.sneck.cm = self.dendrite.cm
        # self.sneck[i].insert('cad_mod_for_ltp')
        # h.taur_cad_mod_for_ltp = 14

        self.shead.insert('Leak_pyr')
        self.shead.e_Leak_pyr = self.dendrite.e_Leak_pyr
        self.shead.gmax_Leak_pyr = self.dendrite.gmax_Leak_pyr
        self.shead.Ra = 100
        self.shead.cm = self.dendrite.cm
        # self.shead[i].insert('cad_mod_for_ltp')
        # h.taur_cad_mod_for_ltp = 14
        # h.depth_cad_mod_for_ltp = h.shead_diam/2

        # print "loc of spine: ", xloc + (i*relative_dist_bw_spines)

        #self.sneck[i].connect(self.dendrite(xloc + (i*relative_dist_bw_spines)), 0)
        self.sneck.connect(self.dendrite(xloc), 0)
        self.shead.connect(self.sneck(1), 0)


    def set_ampa_nmda(self, dend_loc):
        """Used in ObliqueIntegrationTest"""

        ndend, xloc = dend_loc

        exec("self.dendrite=h." + ndend)


        if self.AMPA_name: # if this is given, the AMPA model defined in a mod file is used, else the built in Exp2Syn
            exec("self.ampa = h."+self.AMPA_name+"(0.5, sec=self.shead)")
        else:
            self.ampa = h.Exp2Syn(0.5, sec=self.shead)
            self.ampa.tau1 = self.AMPA_tau1
            self.ampa.tau2 = self.AMPA_tau2
            #print 'The built in Exp2Syn is used as the AMPA component. Tau1 = ', self.AMPA_tau1, ', Tau2 = ', self.AMPA_tau2 , '.'

        if self.NMDA_name: # if this is given, the NMDA model defined in a mod file is used, else the default NMDA model of HippoUnit
            exec("self.nmda= h."+self.NMDA_name+"(0.5, sec=self.shead)")
        else:
            try:
                exec("self.nmda = h."+self.default_NMDA_name+"(0.5, sec=self.shead)")
            except:
                h.nrn_load_dll(self.default_NMDA_path + self.libpath)
                exec("self.nmda = h."+self.default_NMDA_name+"(0.5, sec=self.shead)")

        self.ndend = ndend
        self.xloc = xloc

    def set_netstim_netcon(self, AMPA_weight):
        """Used in ObliqueIntegrationTest"""

        self.ns = h.NetStim()
        self.ns.number = 1
        self.ns.start = self.start

        self.ampa_nc = h.NetCon(self.ns, self.ampa, 0, 0, 0)
        self.nmda_nc = h.NetCon(self.ns, self.nmda, 0, 0, 0)

        self.ampa_nc.weight[0] = AMPA_weight
        self.nmda_nc.weight[0] =AMPA_weight/self.AMPA_NMDA_ratio


    def run_syn(self, dend_loc, weight):
        """Used in ObliqueIntegrationTest"""

        # self.ampa_list = [None] * number
        # self.nmda_list = [None] * number
        # self.ns_list = [None] * number
        # self.ampa_nc_list = [None] * number
        # self.nmda_nc_list = [None] * number

        ndend, xloc = dend_loc

        dend_num = ndend.split('[')[1]  # to get the number of the dendrite (eg. 80 from dendrite[80])
        dend_num = int(dend_num[:-1])
        # print dend_num

        self.initialise()

        exec("self.dendrite=h." + ndend)

        if self.cvode_active:
            h.cvode_active(1)
        else:
            h.cvode_active(0)


        #correcting F factor on the given dendrite taking the number of added spines into consideration
        length = self.dendrite.L

        # print "cm before", self.dendrite.cm
        h.F_factor_correction_with_spines(dend_num, 1/length) # spine density = num of spines / length of dendritic section
        #print "cm after", self.dendrite.cm


        self.create_single_spine(dend_loc)

        self.set_ampa_nmda(dend_loc)

        self.set_netstim_netcon(weight)


        exec("self.sect_loc=h." + str(self.soma)+"("+str(0.5)+")")

        # initiate recording
        rec_t = h.Vector()
        rec_t.record(h._ref_t)

        rec_v = h.Vector()
        rec_v.record(self.sect_loc._ref_v)

        rec_v_dend = h.Vector()
        # rec_v_dend.record(self.shead[0](0.5)._ref_v)
        rec_v_dend.record(self.dendrite(self.xloc)._ref_v)

        h.stdinit()

        dt = 0.025
        h.dt = dt
        h.steps_per_ms = 1 / dt
        h.v_init = self.v_init #-80

        h.celsius = self.celsius
        h.init()
        h.tstop =650
        h.run()

        # get recordings
        t = numpy.array(rec_t)
        v = numpy.array(rec_v)
        v_dend = numpy.array(rec_v_dend)

        return t, v, v_dend


    """ THETA STIMULUS (Takahashi & Magee 2009) """ 

    def create_spine_multiple_loc_theta(self, dend_loc, pathway):

        # dist_bw_spines = 0.284 # um 
        # length = self.dendrite.L
        # relative_dist_bw_spines = dist_bw_spines / length
        # print "length", length
        # print 'relative dist of spines: ', relative_dist_bw_spines

        self.spine_dict.update({'sneck_'+pathway : [h.Section(name='sneck_'+pathway+'[%d]' % i) for i in xrange(len(dend_loc))]}) 
        self.spine_dict.update({'shead_'+pathway : [h.Section(name='shead_'+pathway+'[%d]' % i) for i in xrange(len(dend_loc))]})
        

        #print "e_Leak_pyr", self.dendrite.e_Leak_pyr
        #print "gmax_Leak_pyr", self.dendrite.gmax_Leak_pyr
        #print "cm", self.dendrite.cm

        for i in range(len(dend_loc)):

            ndend, xloc = dend_loc[i]
            exec("dend=h." + ndend)


            self.spine_dict['sneck_'+pathway][i].L = h.sneck_len # 1.58
            self.spine_dict['sneck_'+pathway][i].diam = h.sneck_diam # 0.077
            self.spine_dict['shead_'+pathway][i].L = h.shead_len #0.5 
            self.spine_dict['shead_'+pathway][i].diam = h.shead_diam #0.5

            self.spine_dict['sneck_'+pathway][i].insert('Leak_pyr')
            self.spine_dict['sneck_'+pathway][i].e_Leak_pyr = dend.e_Leak_pyr
            self.spine_dict['sneck_'+pathway][i].gmax_Leak_pyr = dend.gmax_Leak_pyr
            self.spine_dict['sneck_'+pathway][i].Ra = 100
            self.spine_dict['sneck_'+pathway][i].cm = dend.cm
            # self.sneck[i].insert('cad_mod_for_ltp')
            # h.taur_cad_mod_for_ltp = 14

            self.spine_dict['shead_'+pathway][i].insert('Leak_pyr')
            self.spine_dict['shead_'+pathway][i].e_Leak_pyr = dend.e_Leak_pyr
            self.spine_dict['shead_'+pathway][i].gmax_Leak_pyr = dend.gmax_Leak_pyr
            self.spine_dict['shead_'+pathway][i].Ra = 100
            self.spine_dict['shead_'+pathway][i].cm = dend.cm
            # self.shead[i].insert('cad_mod_for_ltp')
            # h.taur_cad_mod_for_ltp = 14
            # h.depth_cad_mod_for_ltp = h.shead_diam/2


            #self.sneck[i].connect(self.dendrite(xloc + (i*relative_dist_bw_spines)), 0)
            self.spine_dict['sneck_'+pathway][i].connect(dend(xloc), 0)
            self.spine_dict['shead_'+pathway][i].connect(self.spine_dict['sneck_'+pathway][i](1), 0)
			

    def set_ampa_nmda_multiple_loc_theta(self, dend_loc, pathway):
        """Used in ObliqueIntegrationTest"""

        # ndend, xloc, loc_type = dend_loc

        # exec("self.dendrite=h." + ndend)


        for i in range(len(dend_loc)):


            if self.AMPA_name: # if this is given, the AMPA model defined in a mod file is used, else the built in Exp2Syn
                exec("self.synapse_lists[\'ampa_list_"+ pathway + "\'][i] = h."+self.AMPA_name+"(0.5, sec=self.spine_dict[\'shead_"+pathway+"\'][i])")
            else: 
                self.synapse_lists['ampa_list_'+pathway][i] = h.Exp2Syn(0.5, sec=self.spine_dict['shead_'+pathway][i])
                self.synapse_lists['ampa_list_'+pathway][i].tau1 = self.AMPA_tau1
                self.synapse_lists['ampa_list_'+pathway][i].tau2 = self.AMPA_tau2
                #print 'The built in Exp2Syn is used as the AMPA component. Tau1 = ', self.AMPA_tau1, ', Tau2 = ', self.AMPA_tau2 , '.'

            if self.NMDA_name: # if this is given, the NMDA model defined in a mod file is used, else the default NMDA model of HippoUnit
                # exec("self.nmda_list[i] = h."+self.NMDA_name+"(0.5, sec=self.shead[i])")
                exec("self.synapse_lists[\'nmda_list_"+ pathway + "\'][i] = h."+self.NMDA_name+"(0.5, sec=self.spine_dict[\'shead_"+pathway+"\'][i])")
            else:
                try:
                    exec("self.synapse_lists[\'nmda_list_"+ pathway + "\'][i] = h."+self.default_NMDA_name+"(0.5, sec=self.spine_dict[\'shead_"+pathway+"\'][i])")
                except:
                    h.nrn_load_dll(self.default_NMDA_path + self.libpath)
                    # neuron.load_mechanisms(self.default_NMDA_path)
                    exec("self.synapse_lists[\'nmda_list_"+ pathway + "\'][i] = h."+self.default_NMDA_name+"(0.5, sec=self.spine_dict[\'shead_"+pathway+"\'][i])")

        # self.ndend = ndend
        # self.xloc = xloc


    def set_netstim_netcon_multiple_loc_theta(self, dend_loc, AMPA_weight, pathway, interval_bw_trains, interval_bw_stimuli_in_train, num_trains, num_stimuli_in_train):
        """Used in ObliqueIntegrationTest"""


        for j in range(num_trains):
            for i in range(len(dend_loc)):
                """
                self.ns_list[j][i] = h.NetStim()
                self.ns_list[j][i].number = 5
                self.ns_list[j][i].interval = 10     # ms 
                self.ns_list[j][i].start = self.start + j * interval_bw_trains 

                self.ampa_nc_list[j][i] = h.NetCon(self.ns_list[j][i], self.ampa_list[i], 0, 0, 0)
                self.nmda_nc_list[j][i] = h.NetCon(self.ns_list[j][i], self.nmda_list[i], 0, 0, 0)

                self.ampa_nc_list[j][i].weight[0] = AMPA_weight
                self.nmda_nc_list[j][i].weight[0] =AMPA_weight/self.AMPA_NMDA_ratio
                """
                self.synapse_lists['ns_list_'+pathway][j][i] = h.NetStim()
                self.synapse_lists['ns_list_'+pathway][j][i].number = num_stimuli_in_train
                self.synapse_lists['ns_list_'+pathway][j][i].interval = interval_bw_stimuli_in_train    # ms 
                self.synapse_lists['ns_list_'+pathway][j][i].start = self.start + j * interval_bw_trains 

                self.synapse_lists['ampa_nc_list_'+pathway][j][i] = h.NetCon(self.synapse_lists['ns_list_'+pathway][j][i], self.synapse_lists['ampa_list_'+pathway][i], 0, 0, 0)
                self.synapse_lists['nmda_nc_list_'+pathway][j][i] = h.NetCon(self.synapse_lists['ns_list_'+pathway][j][i], self.synapse_lists['nmda_list_'+pathway][i], 0, 0, 0)

                self.synapse_lists['ampa_nc_list_'+pathway][j][i].weight[0] = AMPA_weight
                self.synapse_lists['nmda_nc_list_'+pathway][j][i].weight[0] =AMPA_weight/self.AMPA_NMDA_ratio


    def activate_theta_stimuli(self, dend_loc, AMPA_weight, pathway, interval_bw_trains, interval_bw_stimuli_in_train, num_trains, num_stimuli_in_train):


        # self.ampa_list = [None] * len(dend_loc)
        # self.nmda_list = [None] * len(dend_loc)
        # self.ns_list = [None] * len(dend_loc)
        # self.ampa_nc_list = [None] * len(dend_loc)
        # self.nmda_nc_list = [None] * len(dend_loc)
        # self.ampa_nc_list = [[None]*len(dend_loc) for i in range(num_of_trains)]
        # self.nmda_nc_list = [[None]*len(dend_loc) for i in range(num_of_trains)]
        # self.ns_list = [[None]*len(dend_loc) for i in range(num_of_trains)]
        self.synapse_lists.update({'ampa_list_' + pathway : [None] * len(dend_loc),
                            'nmda_list_' + pathway : [None] * len(dend_loc),
                            'ampa_nc_list_' + pathway : [[None]*len(dend_loc) for i in range(num_trains)],
                            'nmda_nc_list_' + pathway : [[None]*len(dend_loc) for i in range(num_trains)],
                            'ns_list_' + pathway : [[None]*len(dend_loc) for i in range(num_trains)] 
                            })  # if synapses of one of the pathways exist already, the dictionary shouldn't be overwritten, but new items are added, therefore 'update' is used. 

        # self.block_Na()

        self.create_spine_multiple_loc_theta(dend_loc, pathway)
        self.set_ampa_nmda_multiple_loc_theta(dend_loc, pathway)
        self.set_netstim_netcon_multiple_loc_theta(dend_loc, AMPA_weight, pathway, interval_bw_trains, interval_bw_stimuli_in_train, num_trains, num_stimuli_in_train)


        """Blocking AMPA component"""
        """
        # self.ampa_nc.weight[0] = 0.0
        for j in range(num_of_trains):
            for i in range(len(dend_loc)):
                self.ampa_nc_list[j][i].weight[0] = 0.0
        """



    def run_simulation(self, dend_loc, recording_loc, tstop):
        """Used in PathwayInteraction Test"""

        (rec_ndend, xloc), distance = recording_loc

        exec("dendrite=h." + rec_ndend)

        if self.cvode_active:
            h.cvode_active(1)
        else:
            h.cvode_active(0)

        dend_sections = []

        for d in dend_loc:
            dend_sections.append(d[0])  

        occurence_of_dend_sections = collections.Counter(dend_sections)    # it is possible that on one dendrite there will be more synapses at different segmenst, this is needed to be taken into account, when recalculating F factor 

        for ndend, num in occurence_of_dend_sections.iteritems():

            dend_num = ndend.split('[')[1]  # to get the number of the dendrite (eg. 80 from dendrite[80])
            dend_num = int(dend_num[:-1])
            # print dend_num

            exec("dend=h." + ndend)
            #correcting F factor on the given dendrite taking the number of added spines into consideration
            length = dend.L

            # print "cm before", self.dendrite.cm
            h.F_factor_correction_with_spines(dend_num, num/length) # spine density = num of spines / length of dendritic section
            #print "cm after", self.dendrite.cm

        exec("self.sect_loc=h." + str(self.soma)+"("+str(0.5)+")")



        # initiate recording
        rec_t = h.Vector()
        rec_t.record(h._ref_t)

        rec_v = h.Vector()
        rec_v.record(self.sect_loc._ref_v)

        rec_v_dend = h.Vector()
        rec_v_dend.record(dendrite(xloc)._ref_v)

        '''
        rec_i_VClamp = h.Vector()
        rec_i_VClamp.record(self.VClamp_stim._ref_i)

        rec_v_shead = h.Vector()
        rec_v_shead.record(self.shead(0.5)._ref_v)

        rec_i_NMDA = h.Vector()
        rec_i_NMDA.record(self.nmda._ref_i)

        rec_ica_NMDA = h.Vector()
        rec_ica_NMDA.record(self.nmda._ref_ica)

        rec_g_NMDA = h.Vector()
        rec_g_NMDA.record(self.nmda._ref_g)
        '''
        h.stdinit()
        dt = 0.025
        h.dt = dt
        h.steps_per_ms = 1 / dt
        h.v_init = self.v_init
        h.celsius = self.celsius
        h.init()
        h.tstop = tstop # 1600
        h.run()
        # get recordings
        t = numpy.array(rec_t)
        v = numpy.array(rec_v)
        v_dend = numpy.array(rec_v_dend)
        # i_VClamp = numpy.array(rec_i_VClamp)
        '''
        v_shead = numpy.array(rec_v_shead)
        i_NMDA = numpy.array(rec_i_NMDA)
        ica_NMDA = numpy.array(rec_ica_NMDA)
        g_NMDA = numpy.array(rec_g_NMDA)
        '''

        """
        print(self.VClamp_stim.gain)
        print(self.VClamp_stim.rstim)
        print(self.VClamp_stim.tau1)
        print(self.VClamp_stim.tau2)
        print(self.VClamp_stim.e0)
        print(self.VClamp_stim.vo0)
        """

        return t, v, v_dend  # , i_VClamp  , v_shead, i_NMDA, ica_NMDA, g_NMDA

    def num_of_possible_locations(self):

        self.initialise()
        locations = [] 

        if self.template_name is not None:
            exec('dendrites=h.testcell.' + self.SecList)

        else:
             exec('dendrites=h.' + self.SecList)

        dendrites = list(dendrites)

        for sec in dendrites:
            for seg in sec:
                locations.append([sec.name(), seg.x])

        return len(locations)

class ModelLoader_Spine_syn_CA1_burst(ModelLoader_Spine_syn):

    def __init__(self, name="model", mod_files_path=None):
        """ Constructor. """
        """ This class should be used with Jupyter notebooks"""
        super(ModelLoader_Spine_syn_CA1_burst, self).__init__(name=name, mod_files_path=mod_files_path)


        self.start=400

        self.SecList = None

    def run_syn(self, dend_loc, weight):
        """Used in ObliqueIntegrationTest"""

        # self.ampa_list = [None] * number
        # self.nmda_list = [None] * number
        # self.ns_list = [None] * number
        # self.ampa_nc_list = [None] * number
        # self.nmda_nc_list = [None] * number

        ndend, xloc = dend_loc

        dend_num = ndend.split('[')[1]  # to get the number of the dendrite (eg. 80 from dendrite[80])
        dend_num = int(dend_num[:-1])
        # print dend_num

        self.initialise()

        exec("self.dendrite=h." + ndend)

        if self.cvode_active:
            h.cvode_active(1)
        else:
            h.cvode_active(0)

        '''
        #correcting F factor on the given dendrite taking the number of added spines into consideration
        length = self.dendrite.L

        # print "cm before", self.dendrite.cm
        h.F_factor_correction_with_spines(dend_num, 1/length) # spine density = num of spines / length of dendritic section
        #print "cm after", self.dendrite.cm
        '''


        self.create_single_spine(dend_loc)

        self.set_ampa_nmda(dend_loc)

        self.set_netstim_netcon(weight)


        exec("self.sect_loc=h." + str(self.soma)+"("+str(0.5)+")")

        # initiate recording
        rec_t = h.Vector()
        rec_t.record(h._ref_t)

        rec_v = h.Vector()
        rec_v.record(self.sect_loc._ref_v)

        rec_v_dend = h.Vector()
        # rec_v_dend.record(self.shead[0](0.5)._ref_v)
        rec_v_dend.record(self.dendrite(self.xloc)._ref_v)

        h.stdinit()

        dt = 0.025
        h.dt = dt
        h.steps_per_ms = 1 / dt
        h.v_init = self.v_init #-80

        h.celsius = self.celsius
        h.init()
        h.tstop =650
        h.run()

        # get recordings
        t = numpy.array(rec_t)
        v = numpy.array(rec_v)
        v_dend = numpy.array(rec_v_dend)

        return t, v, v_dend


    def run_simulation(self, dend_loc, recording_loc, tstop):
        """Used in PathwayInteraction Test"""

        (rec_ndend, xloc), distance = recording_loc

        exec("dendrite=h." + rec_ndend)

        if self.cvode_active:
            h.cvode_active(1)
        else:
            h.cvode_active(0)

        dend_sections = []
        '''
        for d in dend_loc:
            dend_sections.append(d[0])  

        occurence_of_dend_sections = collections.Counter(dend_sections)    # it is possible that on one dendrite there will be more synapses at different segmenst, this is needed to be taken into account, when recalculating F factor 

        for ndend, num in occurence_of_dend_sections.iteritems():

            dend_num = ndend.split('[')[1]  # to get the number of the dendrite (eg. 80 from dendrite[80])
            dend_num = int(dend_num[:-1])
            # print dend_num

            exec("dend=h." + ndend)
            #correcting F factor on the given dendrite taking the number of added spines into consideration
            length = dend.L

            # print "cm before", self.dendrite.cm
            h.F_factor_correction_with_spines(dend_num, num/length) # spine density = num of spines / length of dendritic section
            #print "cm after", self.dendrite.cm
        '''

        exec("self.sect_loc=h." + str(self.soma)+"("+str(0.5)+")")



        # initiate recording
        rec_t = h.Vector()
        rec_t.record(h._ref_t)

        rec_v = h.Vector()
        rec_v.record(self.sect_loc._ref_v)

        rec_v_dend = h.Vector()
        rec_v_dend.record(dendrite(xloc)._ref_v)

        h.stdinit()
        dt = 0.025
        h.dt = dt
        h.steps_per_ms = 1 / dt
        h.v_init = self.v_init
        h.celsius = self.celsius
        h.init()
        h.tstop = tstop # 1600
        h.run()
        # get recordings
        t = numpy.array(rec_t)
        v = numpy.array(rec_v)
        v_dend = numpy.array(rec_v_dend)
        # i_VClamp = numpy.array(rec_i_VClamp)



        return t, v, v_dend  # , i_VClamp  , v_shead, i_NMDA, ica_NMDA, g_NMDA

