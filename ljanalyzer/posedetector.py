'''
Module that abstracts the mediapipe framework and provides a convenient way to 
perform pose detection on images, videos or live streams.

Author: Jakob Faust (software_jaf@mx442.de)
Date: 2023-10-19
'''

import os

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from utils.exception import FileNotFoundException
from .eval import EvalType, Input

class PoseDetector:
    '''
    Abstraction of a pose detector to perform pose detection on images, videos
    or live streams.
    Depending on the desired the detection accuracy, the pose detector can 
    process between 5 and 45 fps.
    The pose detection relies on the mediapipe framework.

    Parameters
    ----------
    input_type : Input
        Specifies the type of input to perform pose detection on.
        Must be one of the following: IMAGE, VIDEO, LIVESTREAM
    eval_type : EvalType
        Specifies the desired accuracy of the pose detection.
        Must be one of the following: REALTIME, NEAR_REALTIME, FULL
    '''
    def __init__(self, input_type: Input, eval_type: EvalType) -> None:
        self.input_type = input_type
        self.eval_type = eval_type
        self.pose_detector = self.__init_pose_detector()

    def get_body_key_points(self, image, timestamp):
        '''
        performs pose detection on the provided input and returns the detected
        body key points.
        input can be an image, video frame or live stream frame.

        Parameters
        ----------
        image : numpy.ndarray
            input to perform pose detection on.
            shape must be (height, width, channels).
        timestamp : float
            timestamp of the input frame.
            can simply be the frame number when using a video file.

        '''
        rtrn = None
        if self.input_type == Input.IMAGE:
            rtrn = self.pose_detector.detect_for_image(image)
        if self.input_type == Input.VIDEO:
            rtrn = self.pose_detector.detect_for_video(image, timestamp)
        if self.input_type == Input.LIVESTREAM:
            rtrn = self.pose_detector.detect_async(image, timestamp)
        return rtrn

    def get_config(self)->tuple:
        '''
        Returns a tuple of the current configuration.

        Returns
        -------
        tuple
            tuple of the current configuration.
            (input type, evaluation type)
        '''
        return self.input_type.name, self.eval_type.name

    def get_input_type(self)->str:
        '''
        Returns the current input type.
        (IMAGE, VIDEO or LIVESTREAM)
        '''
        return self.input_type.name

    def get_eval_type(self)->str:
        '''
        Returns the current evaluation type.
        (REALTIME, NEAR_REALTIME or FULL)
        '''
        return self.eval_type.name

    def __init_pose_detector(self)->None:
        current_path = os.path.dirname(__file__)
        if self.eval_type == EvalType.REALTIME:
            model_path = '../models/pose_landmarker_lite.task'
        if self.eval_type == EvalType.NEAR_REALTIME:
            model_path = '../models/pose_landmarker_full.task'
        if self.eval_type == EvalType.FULL:
            model_path = '../models/pose_landmarker_heavy.task'
        model_path = os.path.join(current_path, model_path)
        if not os.path.exists(model_path):
            raise FileNotFoundException(f"""model file could not be found -
                                        check file path {model_path}""")
        base_options = python.BaseOptions(model_asset_path=model_path)
        vision_running_mode = vision.RunningMode
        running_mode = vision_running_mode.LIVE_STREAM
        if self.input_type == Input.IMAGE:
            running_mode = vision_running_mode.IMAGE
        if self.input_type == Input.VIDEO:
            running_mode = vision_running_mode.VIDEO
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=running_mode)
        return vision.PoseLandmarker.create_from_options(options)
