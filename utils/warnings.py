'''
Module that provides several custom warnings.

Author: Jakob Faust (software_jaf@mx442.de)
Date: 2023-10-124
'''
class PotentialRaceConditionWarning(ResourceWarning):
    '''
    warning is shown if the software detects a risk of a possible race 
    condition
    '''
    pass
