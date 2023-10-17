from .evalType import EvalType, Input
from .exception import FileNotFoundException

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os


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
        rtrn = 0
        if self.input_type == Input.IMAGE:
            rtrn = self.pose_detector.detect_for_image(image)
        if self.input_type == Input.VIDEO:
            rtrn = self.pose_detector.detect_for_video(image, timestamp)
        if self.input_type == Input.LIVESTREAM:
            rtrn = self.pose_detector.detect_async(image, timestamp)
        return rtrn
    
    def __init_pose_detector(self):
        current_path = os.path.dirname(__file__)
        print(current_path)
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
        baseOptions = python.BaseOptions(model_asset_path=model_path)
        VisionRunningMode = vision.RunningMode
        running_mode = VisionRunningMode.LIVE_STREAM
        if self.input_type == Input.IMAGE:
            running_mode = VisionRunningMode.IMAGE
        if self.input_type == Input.VIDEO:
            running_mode = VisionRunningMode.VIDEO
        options = vision.PoseLandmarkerOptions(
            base_options=baseOptions,
            running_mode=running_mode)
        return vision.PoseLandmarker.create_from_options(options)