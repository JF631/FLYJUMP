'''
Module for defining config used for evaluation.
The provided enums are used to specify the desired accuracy of the pose 
detection as well as the type of input to perform pose detection on.

Author: Jakob Faust
Date: 2023-10-17
'''
from enum import Enum

class EvalType(Enum):
    '''
    Defines the desired accuracy of the pose detection.
    Supported evaluation types are: REALTIME, NEAR_REALTIME, FULL

    Usage hint
    ----------
    REALTIME: up to ~45 fps
    NEAR_REALTIME: up to ~20 fps
    FULL: up to ~5 fps
    '''
    REALTIME = 0 # up to ~45 fps
    NEAR_REALTIME = 1 # up to ~20 fps
    FULL = 2 # up to ~5 fps

class Input(Enum):
    '''
    Defines the type of input to perform pose detection on.
    Supported inputs are: IMAGE, VIDEO, LIVESTREAM
    '''
    IMAGE = 0
    VIDEO = 1
    LIVESTREAM = 2
