from .evalType import EvalType

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os


class PoseDetector:
    def __init__(self, evalType: EvalType) -> None:
        self.evalType = evalType
        self.pose_detector = self._init_pose_detector()
        
    def get_body_key_points(self, image, timestamp):
        rtrn = 0
        if self.evalType == EvalType.IMAGE:
            rtrn = self.pose_detector.detect_for_image(image)
        if self.evalType == EvalType.VIDEO:
            rtrn = self.pose_detector.detect_for_video(image, timestamp)
        if self.evalType == EvalType.LIVESTREAM:
            rtrn = self.pose_detector.detect_async(image, timestamp)
        return rtrn
    
    def _init_pose_detector(self):
        model_path = 'C:/Users/Jakob/FLYJUMP/models/pose_landmarker_full.task'
        if not os.path.exists(model_path):
            print("not exits")
        baseOptions = python.BaseOptions(model_asset_path=model_path)
        VisionRunningMode = vision.RunningMode
        running_mode = VisionRunningMode.LIVE_STREAM
        if self.evalType == EvalType.IMAGE:
            running_mode = VisionRunningMode.IMAGE
        if self.evalType == EvalType.VIDEO:
            running_mode = VisionRunningMode.VIDEO
        options = vision.PoseLandmarkerOptions(
            base_options=baseOptions,
            running_mode=running_mode)
        return vision.PoseLandmarker.create_from_options(options)