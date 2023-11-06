'''
Module that provides usefull methods to use numpy frames from opencv in 
combination with mediapipe

Author: Jakob Faust (software_jaf@mx442.de)
Date: 2023-10-22
'''

import cv2
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np

class Frame():
    '''
    Abstraction of a opencv / numpy video frame of shape 
    (height, width, channels).

    Provides methods to annotate a frame object with given pose landmarks and 
    transform the frame data to mediapipe image format.

    '''
    def __init__(self, frame:np.ndarray) -> None:
        self.__data = frame
        self.__right_knee_angle = 0.0
        self.__left_knee_angle = 0.0
        self.__hip_height = 0.0
        self.__foot_tip = 0.0

    def __bool__(self):
        return self.__data is not None

    def annotate(self, pose_landmarks, as_overlay=True) -> None:
        '''
        Visualizes detected pose key points on the current frame.
        
        Parameters
        -----------
        pose_landmarks : list 
                detection result from mediapipe (pass 
                result.pose_landmarks as argument)
        as_overlay : bool
                if true, the detected pose is drawn over the original frame,
                otherwise only the keypoints are drawn on black background
        '''
        if not as_overlay:
            self.__data = np.zeros_like(self.__data)
        for pose in pose_landmarks:
            hip, knee, foot = pose[24:30:2]
            self.foot_tip = pose[32]
            self.__hip_height = hip.y
            hip_knee_vec = np.array([knee.x - hip.x, knee.y - hip.y])
            knee_foot_vec = np.array([foot.x - knee.x, foot.y - knee.y])
            self.__right_knee_angle = 180 - np.rad2deg(
                np.arccos((np.vdot(hip_knee_vec, knee_foot_vec)) /
                (np.linalg.norm(hip_knee_vec) *
                 np.linalg.norm(knee_foot_vec))))
            cv2.putText(self.__data, str(self.__right_knee_angle), (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            pose_proto = landmark_pb2.NormalizedLandmarkList()
            pose_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(
                    x=landmark.x, y=landmark.y, z=landmark.z)
                    for landmark in pose
            ])
            solutions.drawing_utils.draw_landmarks(
                self.__data,
                pose_proto,
                solutions.pose.POSE_CONNECTIONS,
                solutions.drawing_styles.get_default_pose_landmarks_style()
            )

    def to_mediapipe_image(self) -> mp.Image:
        '''
        Converts numpy frame (BGR) to mediapipe image object (SRGB).
        Mediapipe can only handle this image format!

        Returns
        -------
        frame : mp.Image
            current frame in Mediapipe SRGB Image format
        '''
        return mp.Image(image_format=mp.ImageFormat.SRGB,
                                    data=self.__data)

    def data(self) -> np.ndarray:
        '''
        Returns the frame in numpy format.

        Returns
        -------
        frame : np.ndarray
            current frame data, shape (height, width, channels).
        '''
        return self.__data

    def knee_angles(self) -> np.ndarray:
        '''
        Returns
        -------
        knee_angles : np.ndarray
            knee_angles[0]: right knee angle, knee_angles[1]: left knee angle´.
        '''
        return np.array([self.__right_knee_angle, self.__left_knee_angle])
    
    def foot_pos(self):
        return self.foot_tip

    def centroid_height(self) -> float:
        '''
        Returns
        -------
        centroid_height : float
            relative height of body centroid in current frame. 
        '''
        return self.__hip_height
