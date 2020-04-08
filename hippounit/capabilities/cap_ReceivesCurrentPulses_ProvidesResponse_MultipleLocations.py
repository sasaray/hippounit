import sciunit
from sciunit import Capability


class ReceivesCurrentPulses_ProvidesResponse_MultipleLocations(sciunit.Capability):
    """Indicates that current pulses of given frequency can be injected into the model as
    a square pulse. And records at multiple locations."""


    def inject_current_pulses_record_respons_multiple_loc(self, amp, delay, dur_of_pulse, dur_of_stim,  num_of_pulses, frequency, section_stim, loc_stim, dend_locations):
        """This function must be implemented by the model.

        Must return numpy arrays containing the time vector and the voltage values recorded on the stimulus location (soma), and a nested dictionary containing voltage vectors of the recorded dendritic locations at the examined distances in this test.

        eg.: {dist1: { ('trunk_segment1',location1): numpy.array(voltage trace),
                       ('trunk_segment2',location2): numpy.array(voltage trace)
                      },
                      { ('trunk_segment3',location3): numpy.array(voltage trace),
                       ('trunk_segment4',location4): numpy.array(voltage trace)
                      },
        Or return the dictionary without the outer key, if the input dend_locations is a list of locations, not a dictionary with distances as keys.  This is because the ProvidesRecordingLocationsOnTrunk and the ProvidesRandomDendriticLocations capabilities returns the dendritic locations in different formats.

        amp: amplitude of the current injection  pulses (nA)

        delay: delay before the first current pulse (ms)

        dur of pulse : duration of the current pulses

        num_of_pulses : number of the pulses

        dur_of_sim : duration of the simulation

        frequency : frequency of the pulses

        section_stim: string - the name of the stimulated section (eg. "soma")

        loc_stim: float - location on the stimulated section (eg. 0.5)

        dend_locations: dict of dendritic location with distance range as a key: dend_loc = (dist1, ['trunk_segment1_1',location],['trunk_segment1_2',location]), (dist2, ['trunk_segment2',location]),(dist3, ['trunk_segment3',location]), (dist4, ['trunk_segment4',location]) 
        
       or list of recording locations in the form: [['trunk_segment1',location1],['trunk_segment2',location2],['trunk_segment3',location3]]
       """
        raise NotImplementedError()

    def inject_current_pulses_get_multiple_vm(self, amp, delay, dur_of_pulse, dur_of_stim, num_of_pulses, frequency, section_stim, loc_stim, dend_locations):
        # v : dictionary -  keys: dendritic location, values: the voltage trace for each recording locations
        t, v_stim, v = self. inject_current_pulses_record_respons_multiple_loc(amp, delay, dur_of_pulse, dur_of_stim, num_of_pulses, frequency, section_stim, loc_stim, dend_locations)
        return t, v_stim, v
