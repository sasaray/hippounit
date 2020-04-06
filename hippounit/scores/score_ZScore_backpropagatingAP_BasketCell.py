from __future__ import division
from builtins import str
from builtins import range

from sciunit import Score
import numpy
from sciunit.utils import assert_dimensionless
import collections

class ZScore_backpropagatingAP_BasketCell(Score):
    """
    Average of Z scores. A float indicating the average of standardized difference
    from reference means for back-propagating AP amplitudes, rise slope and half-duration for Basket Cell .
    """

    def __init__(self, score, related_data={}):

        if not isinstance(score, Exception) and not isinstance(score, float):
            raise InvalidScoreError("Score must be a float.")
        else:
            super(ZScore_backpropagatingAP_BasketCell,self).__init__(score, related_data=related_data)

    @classmethod
    def compute(cls, observation, prediction):
        """Computes average of z-scores from observation and prediction for back-propagating AP amplitudes"""

        errors = {'soma' : {} ,
                  'apical' :{},
                  'basal' :{}
                  } 

        feature_errors=numpy.array([])

        for k, v in observation['soma'].items():
            p_value = prediction['soma'][k]
            o_mean = v['mean'] 
            o_std = v['std']

            try:
                error = abs(p_value - o_mean)/o_std
                error = assert_dimensionless(error)
            except (TypeError,AssertionError) as e:
                error = e
            errors['soma'].update({k : error})
            feature_errors=numpy.append(feature_errors, error)

        for key, value in observation['apical'].items():
            errors['apical'].update({int(key):{}})
            for k, v in value.items():
                p_value = prediction['apical'][int(key)][k]['mean'] 
                o_mean = v['mean'] 
                o_std = v['std']
                # print(p_value, o_mean, o_std)
                try:
                   error = abs(p_value - o_mean)/o_std
                   error = assert_dimensionless(error)
                except (TypeError,AssertionError) as e:
                    error = e
                errors['apical'][int(key)].update({k : error})
                feature_errors=numpy.append(feature_errors, error)
 
        for key, value in observation['basal'].items():
            errors['basal'].update({int(key):{}})
            for k, v in value.items():
                p_value = prediction['basal'][int(key)][k]['mean'] 
                o_mean = v['mean'] 
                o_std = v['std']
                try:
                   error = abs(p_value - o_mean)/o_std
                   error = assert_dimensionless(error)
                except (TypeError,AssertionError) as e:
                    error = e
                errors['basal'][int(key)].update({k : error})
                feature_errors=numpy.append(feature_errors, error)
        score_avg=numpy.nanmean(feature_errors)

        return score_avg, errors

    def __str__(self):

        return 'ZScore_avg = %.2f' % self.score
