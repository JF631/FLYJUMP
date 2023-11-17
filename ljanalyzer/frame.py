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

    Usage
    -----
    Objects of this class are re-useable.
    To update the data in a frame object simply call frame.update(new_frame)

    If you want to make sure no data from the old frame is re-used, call
    frame.clear() before frame.update(new_frame) 

    '''
    def __init__(self, frame: np.ndarray = None) -> None:
        self.__right_knee_angle = 0.0
        self.__left_knee_angle = 0.0
        self.__hip_position = np.empty(2)
        self.__data = None
        self.foot_positions: tuple = None
        self.dims = (0, 0, 0) # (height, width, channels)
        if frame is not None:
            self.update(frame)
            self.dims = self.__data.shape

    def __bool__(self):
        return self.__data is not None

    def update(self, frame:np.ndarray)->None:
        '''
        replaces frame data in the current frame object.

        Parameters
        ----------
        frame : np.ndarray
            new frame of shape (height, width, channels)
        '''
        self.__data = frame
        self.dims = self.__data.shape

    def clear(self):
        '''
        clears current frame.
        '''
        self.__data = None

    def __calc_knee_angle(self, key_points: tuple)->float:
        '''
        calculates knee angle.

        Parameters
        ----------
        key_points : tuple (hip, knee, foot)
            key_points from pose detection.
            Must contain hip, knee and foot values, each of which must have
            .x and .y values 

        Returns
        -------
        knee_angle : float
            knee angle in degrees.
        '''
        hip, knee, foot = key_points
        hip_knee_vec = np.array(
            [
                knee.x - hip.x,
                knee.y - hip.y
            ], dtype='f4')
        knee_foot_vec = np.array(
            [
                foot.x - knee.x,
                foot.y - knee.y
            ], dtype='f4')
        return 180 - np.rad2deg(
            np.arccos((np.vdot(hip_knee_vec, knee_foot_vec)) /
            (np.linalg.norm(hip_knee_vec) *
            np.linalg.norm(knee_foot_vec))))

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
            self.foot_positions = pose[-2:]
            self.__hip_position = np.array([pose[24].x, pose[24].y])
            self.__right_knee_angle = self.__calc_knee_angle(pose[24:30:2])
            self.__left_knee_angle = self.__calc_knee_angle(pose[23:29:2])
            cv2.putText(self.__data, f"""right: {self.__right_knee_angle:.4f}
                        left:{self.__left_knee_angle:.4f}""", (10, 30),
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

    def to_rgb(self) -> np.ndarray:
        return cv2.cvtColor(self.__data, cv2.COLOR_BGR2RGB)

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
            knee_angles[0]: right knee angle, knee_angles[1]: left knee angle.
        '''
        return np.array([self.__right_knee_angle, self.__left_knee_angle],
            dtype='f4')

    def foot_pos(self):
        '''
        Foot position matrix.

        Returns
        -------
        foot_pos : np.ndarray
            Foot position matrix of shape (2,2)

        Usage
        -----
        The matrix is ordered as follows:

        [[left_foot.x, right_foot.x],
         [left_foot.y, right_foot.y]]
        '''
        return np.array([
            [self.foot_positions[0].x, self.foot_positions[1].x],
            [self.foot_positions[0].y, self.foot_positions[1].y]
        ])

    def centroid_height(self) -> float:
        '''
        Returns
        -------
        centroid_height : float
            normalized height of body centroid in current frame. 
        '''
        return self.__hip_position[1]
    
    def hip_pos(self)->np.ndarray:
        '''
        Returns
        -------
        hip_position : np.ndarray
            normalized hip position in current frame.
            [hip.x, hip.y]
        '''
        return self.__hip_position
