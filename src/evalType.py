from enum import Enum

class EvalType(Enum):
    REALTIME = 0 # up to ~45 fps
    NEAR_REALTIME = 1 # up to ~20 fps 
    FULL = 2 # up to ~5 fps

class Input(Enum):
    IMAGE = 0
    VIDEO = 1
    LIVESTREAM = 2