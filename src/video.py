from .frameBuffer import FrameBuffer
from .poseDetector import PoseDetector
from .evalType import Input, EvalType
from .exception import FileNotFoundException, GeneralException

import cv2
import os
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import threading

class Video():
    '''
    Abstraction of a video file. 
    Main purpose is to provide a convenient way to analyze a video regarding
    pose detection.
    
    The analysis is performed within two threads:
        1) Read video frame by frame from file, visualize the frame and add it 
        to a buffer.

        2) Read from the same buffer and perform pose detection.


    Parameters
    ----------
    path : str
        path to video file.
    '''
    def __init__(self, path:str) -> None:
        self.frame_count = 0
        self.frame_rate = 0
        self.dims = (0, 0, 0)
        self.__open(path)
        self.path = path
        self.output_path = ''
        self.detector = PoseDetector(Input.VIDEO, EvalType.FULL)
        self.frame_buffer = FrameBuffer(self.frame_count, self.dims, 
                                        lock=False)
        self.video_completed = threading.Event()

    def __open(self, path:str):
        '''
        Tries to open the video file, read metadata and display the first frame

        Parameters
        ----------
        path : str
            path to video file.
        '''
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise FileNotFoundException(f"""file could not be opened - check
                                        file path {path}""")
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_rate = cap.get(cv2.CAP_PROP_FPS)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.dims = (height, width, 3)
        valid, _ = cap.read()
        if not valid:
            self.video_completed.set()
            raise GeneralException(f"""OpenCV could not read from video file 
                                   {path}""")
        # cv2.imshow("Video", first_frame) 
        cap.release()

        
    def __add_pose_overlay(self, frame, pose_landmarks, as_overlay=True):
        '''
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
        '''
        if as_overlay:
            rtrn = np.copy(frame)
        else:
            rtrn = np.zeros_like(frame)
        for pose in pose_landmarks:
            pose_proto = landmark_pb2.NormalizedLandmarkList()
            pose_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(
                    x=landmark.x, y=landmark.y, z=landmark.z) 
                    for landmark in pose
            ])
            solutions.drawing_utils.draw_landmarks(
                rtrn,
                pose_proto,
                solutions.pose.POSE_CONNECTIONS,
                solutions.drawing_styles.get_default_pose_landmarks_style()
            )
        return rtrn
        
    def __read_video_file(self):
        '''
        reads video frame by frame from file, visualizes the frame and adds it
        to a frame buffer.
        '''
        cap = cv2.VideoCapture(self.path)
        while cap.isOpened():
            valid, frame = cap.read()
            if not valid:
                self.video_completed.set()
                print("video ended or an error occurred")
                break
            self.frame_buffer.add(frame)
            cv2.imshow("Video", frame)
            if (cv2.waitKey(int((1 / self.frame_rate) * 1000)) &
                    0xFF == ord('q')):
                break
        cap.release()
        print("released input")
        
    def __perform_pose_detection(self):
        '''
        Runs pose detection on frames from the frame buffer and writes the
        result to a video file.
        '''
        self.output_path = os.path.dirname(__file__)
        self.output_path = os.path.join(self.output_path, '../output/test.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(self.output_path, fourcc, 30, (self.dims[1], self.dims[0]))
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
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, 
                                    data=frame)
                res = self.detector.get_body_key_points(mp_image, counter)
                if res.pose_landmarks:
                    frame = self.__add_pose_overlay(frame, res.pose_landmarks, 
                                                  as_overlay=False)
                    out.write(frame)
                # cv2.imshow("TEST", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            elif playback and self.video_completed.is_set():
                break
        out.release()
        lost_frames = (1 - (counter / self.frame_count)) * 100
        print(f"lost frames: {self.frame_count - counter} \
              ({lost_frames:.2f}%)" )
        cv2.destroyAllWindows()
        
    def analyze(self):
        load_thread = threading.Thread(target=self.__read_video_file)
        visualize_thread = threading.Thread(
            target=self.__perform_pose_detection)
        load_thread.start()
        visualize_thread.start()
        load_thread.join()
        visualize_thread.join()
    
    def get_path(self):
        return self.path
        
    def get_output_path(self):
        return self.output_path