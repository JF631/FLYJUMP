from .frameBuffer import FrameBuffer
from .poseDetector import PoseDetector
from .evalType import EvalType
from .exception import InsufficientMemoryException, GeneralException

import cv2
import psutil
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import threading

class Video():
    def __init__(self, path:str) -> None:
        self.path = path
        self.output_path = ''
        self.detector = PoseDetector(EvalType.VIDEO)
        self.frame_rate = 0
        self.frame_count = 0
        self.dims = (0, 0, 0)
        self.frame_buffer = FrameBuffer(64, lock=False)
        self.video_completed = threading.Event()
        
    def add_pose_overlay(self, frame, pose_landmarks, as_overlay=True):
        """
        Visualizes detected pose key points on a given frame.
        Within the process a copy of the original frame is generated, annotated 
        and returned  
        
        Parameters
        -----------
        frame : np.array 
                frame in pixel (R,G,B) representation.\n
        pose_landmarks : list 
                detection result from mediapipe (pass 
                result.pose_landmarks as argument)

        Returns
        --------
        np.array : annotated frame with visualized key points of shape 
        (width, height, color_depth)
        """
        if as_overlay:
            rtrn = np.copy(frame)
        else:
            rtrn = np.zeros_like(frame)
        for pose in pose_landmarks:
            pose_proto = landmark_pb2.NormalizedLandmarkList()
            pose_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose
            ])
            solutions.drawing_utils.draw_landmarks(
                rtrn,
                pose_proto,
                solutions.pose.POSE_CONNECTIONS,
                solutions.drawing_styles.get_default_pose_landmarks_style()
            )
        return rtrn
        
    def load(self):
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            print(f"file could not be opened - check file path: {self.path}")
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.dims = (width, height, 3) # default for color images (BGR)
        print(f"width {width}, height {height}")
        print(self.frame_count)
        self.frame_rate = cap.get(cv2.CAP_PROP_FPS)
        while cap.isOpened():
            valid, frame = cap.read()
            if not valid:
                self.video_completed.set()
                print("video ended or an error occurred")
                break
            self.frame_buffer.add(frame)
            cv2.imshow("Video", frame)
            if cv2.waitKey(int((1 / self.frame_rate) * 1000)) & 0xFF == ord('q'):
                break
        cap.release()
        print("released input")
        
    def play(self):
        self.output_path = 'C:/Users/Jakob/FLYJUMP/test.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(self.output_path, fourcc, 30, (1920, 1080))
        playback = False 
        lost_frames = 0
        counter = 0 
        while True:
            if self.frame_buffer.current_frames() > 0:
                frame = self.frame_buffer.pop()
                playback = True
                if frame is None:
                    print("Empty frame received")
                    break
                counter += 1
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                res = self.detector.get_body_key_points(mp_image, counter)
                if res.pose_landmarks:
                    frame = self.add_pose_overlay(frame, res.pose_landmarks, as_overlay=False)
                    out.write(frame)
                # cv2.imshow("TEST", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            elif playback and self.video_completed.is_set():
                break
        
        out.release()
        lost_frames = (1 - (counter / self.frame_count)) * 100
        print(f"lost frames: {self.frame_count - counter} ({lost_frames:.2f}%)" )
        cv2.destroyAllWindows()
        
    def analyze(self):
        load_thread = threading.Thread(target=self.load)
        visualize_thread = threading.Thread(target=self.play)
        load_thread.start()
        visualize_thread.start()
        load_thread.join()
        visualize_thread.join()
    
    def get_path(self):
        return self.path
        
    def get_output_path(self):
        return self.output_path
    
    def add_keypoints(self, as_overlay=True):
        current_frame = 0
        current_index = 0
        while self.frame_buffer:
            current_frame = self.frame_buffer.get_frame(current_index)
            self.detector.get_body_key_points()
            
    def _ensure_memory(self, frame_count, frame_height, frame_width):
        required_memory = frame_count * frame_height * frame_width * 3 # 3 channels (BGR)
        available_memory = psutil.virtual_memory().total
        if required_memory > available_memory:
            print(f"video is too large:  {required_memory / (1024 * 1024)} MB")
            return False
        print(f"memory required: {required_memory / 1024 / 1024} MB")
        return True