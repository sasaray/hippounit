from __future__ import division
from builtins import str
from builtins import range

from sciunit import Score
import numpy
from sciunit.utils import assert_dimensionless
import collections

class ZScore_backpropagatingAP_CA3_PC(Score):
    """
    Average of Z scores. A float indicating the average of standardized difference
    from reference means for back-propagating AP amplitudes, half-duration for CA3 pyramidal cell .
    """

    def __init__(self, score, related_data={}):

        if not isinstance(score, Exception) and not isinstance(score, float):
            raise InvalidScoreError("Score must be a float.")
        else:
            super(ZScore_backpropagatingAP_CA3_PC,self).__init__(score, related_data=related_data)

    @classmethod
    def compute(cls, observation, prediction):
        """Computes average of z-scores from observation and prediction for back-propagating AP amplitudes"""

        errors = {'soma' : {'long input' : {}, 
                            'train of brief inputs' :{}},
                  'apical' : {'long input' : {}, 
                              'train of brief inputs' :{}},
                  'basal' : {'long input' : {}, 
                             'train of brief inputs' :{}}  
                  } 

        feature_errors=numpy.array([])

        for k, v in observation['soma']['long input'].items():
            p_value = prediction['soma']['long input'][k]
            o_mean = v['mean'] 
            o_std = v['std']

            try:
                error = abs(p_value - o_mean)/o_std
                error = assert_dimensionless(error)
            except (TypeError,AssertionError) as e:
                error = e
            errors['soma']['long input'].update({k : error})
            feature_errors=numpy.append(feature_errors, error)

        for key, value in observation['apical']['long input'].items():
            errors['apical']['long input'].update({int(key):{}})
            for k, v in value.items():
                p_value = prediction['apical']['long input'][int(key)][k]['mean'] 
                o_mean = v['mean'] 
                o_std = v['std']
                # print(p_value, o_mean, o_std)
                try:
                   error = abs(p_value - o_mean)/o_std
                   error = assert_dimensionless(error)
                except (TypeError,AssertionError) as e:
                    error = e
                errors['apical']['long input'][int(key)].update({k : error})
                feature_errors=numpy.append(feature_errors, error)
 
        for key, value in observation['basal']['long input'].items():
            errors['basal']['long input'].update({int(key):{}})
            for k, v in value.items():
                p_value = prediction['basal']['long input'][int(key)][k]['mean'] 
                o_mean = v['mean'] 
                o_std = v['std']
                try:
                   error = abs(p_value - o_mean)/o_std
                   error = assert_dimensionless(error)
                except (TypeError,AssertionError) as e:
                    error = e
                errors['basal']['long input'][int(key)].update({k : error})
                feature_errors=numpy.append(feature_errors, error)
        score_avg=numpy.nanmean(feature_errors)


        for freq in list(sorted(observation['soma']['train of brief inputs'].keys())):
            errors['soma']['train of brief inputs'][freq] = collections.OrderedDict()
            errors['apical']['train of brief inputs'][freq] = collections.OrderedDict()
            errors['basal']['train of brief inputs'][freq] = collections.OrderedDict()
            for k, v in observation['soma']['train of brief inputs'][freq].items():
                p_value = prediction['soma']['train of brief inputs'][freq][k]
                o_mean = v['mean'] 
                o_std = v['std']

                try:
                    error = abs(p_value - o_mean)/o_std
                    error = assert_dimensionless(error)
                except (TypeError,AssertionError) as e:
                    error = e
                errors['soma']['train of brief inputs'][freq].update({k : error})
                feature_errors=numpy.append(feature_errors, error)

            for key, value in observation['apical']['train of brief inputs'][freq].items():
                errors['apical']['train of brief inputs'][freq].update({int(key):{}})
                for k, v in value.items():
                    p_value = prediction['apical']['train of brief inputs'][freq][int(key)][k]['mean'] 
                    o_mean = v['mean'] 
                    o_std = v['std']
                    # print(p_value, o_mean, o_std)
                    try:
                        error = abs(p_value - o_mean)/o_std
                        error = assert_dimensionless(error)
                    except (TypeError,AssertionError) as e:
                        error = e
                    errors['apical']['train of brief inputs'][freq][int(key)].update({k : error})
                    feature_errors=numpy.append(feature_errors, error)
 
            for key, value in observation['basal']['train of brief inputs'][freq].items():
                errors['basal']['train of brief inputs'][freq].update({int(key):{}})
                for k, v in value.items():
                    p_value = prediction['basal']['train of brief inputs'][freq][int(key)][k]['mean'] 
                    o_mean = v['mean'] 
                    o_std = v['std']
                    try:
                        error = abs(p_value - o_mean)/o_std
                        error = assert_dimensionless(error)
                    except (TypeError,AssertionError) as e:
                        error = e
                    errors['basal']['train of brief inputs'][freq][int(key)].update({k : error})
                    feature_errors=numpy.append(feature_errors, error)

        return score_avg, errors

    def __str__(self):

        return 'ZScore_avg = %.2f' % self.score
