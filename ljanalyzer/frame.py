"""
Module that provides usefull methods to use numpy frames from opencv in 
combination with mediapipe

Author: Jakob Faust (software_jaf@mx442.de)
Date: 2023-10-22
"""

import cv2
import mediapipe as mp
import numpy as np
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

from .eval import Filter


class Frame:
    """
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

    """

    def __init__(self, frame: np.ndarray = None) -> None:
        self.__right_knee_angle = 0.0
        self.__left_knee_angle = 0.0
        self.__hip_position = np.empty(2)
        self.__data = None
        self.__pre_processed = None
        self.foot_positions: tuple = None
        self.dims = (0, 0, 0)  # (height, width, channels)
        if frame is not None:
            self.update(frame)

    def __bool__(self):
        return self.__data is not None

    def update(self, frame: np.ndarray) -> None:
        """
        replaces frame data in the current frame object.

        Parameters
        ----------
        frame : np.ndarray
            new frame of shape (height, width, channels)
        """
        self.__data = np.asarray(frame)
        self.__pre_processed = np.empty_like(self.__data)
        self.dims = self.__data.shape

    def clear(self):
        """
        clears current frame.
        """
        self.__data = None

    def __calc_knee_angle(self, key_points: tuple) -> float:
        """
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
        """
        hip, knee, foot = key_points
        hip_knee_vec = np.array([knee.x - hip.x, knee.y - hip.y], dtype="f4")
        knee_foot_vec = np.array([foot.x - knee.x, foot.y - knee.y], dtype="f4")
        return 180 - np.rad2deg(
            np.arccos(
                (np.vdot(hip_knee_vec, knee_foot_vec))
                / (np.linalg.norm(hip_knee_vec) * np.linalg.norm(knee_foot_vec))
            )
        )

    def annotate(self, pose_landmarks, as_overlay=True) -> None:
        """
        Visualizes detected pose key points on the current frame.

        Parameters
        -----------
        pose_landmarks : list
                detection result from mediapipe (pass
                result.pose_landmarks as argument)
        as_overlay : bool
                if true, the detected pose is drawn over the original frame,
                otherwise only the keypoints are drawn on black background
        """
        if not as_overlay:
            self.__data = np.zeros_like(self.__data)
        for pose in pose_landmarks:
            self.foot_positions = pose[-2:]
            self.__hip_position = np.array([pose[24].x, pose[24].y])
            self.__right_knee_angle = self.__calc_knee_angle(pose[24:30:2])
            self.__left_knee_angle = self.__calc_knee_angle(pose[23:29:2])
            cv2.putText(
                self.__data,
                f"""right: {self.__right_knee_angle:.4f}
                        left:{self.__left_knee_angle:.4f}""",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )
            pose_proto = landmark_pb2.NormalizedLandmarkList()
            pose_proto.landmark.extend(
                [
                    landmark_pb2.NormalizedLandmark(
                        x=landmark.x, y=landmark.y, z=landmark.z
                    )
                    for landmark in pose
                ]
            )
            solutions.drawing_utils.draw_landmarks(
                self.__data,
                pose_proto,
                solutions.pose.POSE_CONNECTIONS,
                solutions.drawing_styles.get_default_pose_landmarks_style(),
            )

    def pre_process(self, filter: Filter = None, inplace=False):
        """
        applies convolutional filter to the current frame.

        Parameters
        ----------
        filter : Filter
            Highpass, Lowpass or BILATERAL
        inplace : bool
            if True, all filters are applied directly on the current frames'
            data.
            Otherwise, a copy is created and filtered.
        """
        if not filter:
            return
        sharpen_kernel = np.array([[0, -1, 0, -1, 5, -1, 0, -1, 0]], dtype="f4")
        blur_kernel = 1 / 9 * np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype="f4")
        if filter == Filter.LOWPASS:
            self.apply_filter(kernel=blur_kernel, inplace=inplace)
        if filter == Filter.HIGHPASS:
            self.apply_filter(kernel=sharpen_kernel, inplace=inplace)
        if filter == Filter.BILATERAL:
            self.bilateral_filter(output_result=inplace)

    def to_mediapipe_image(self) -> mp.Image:
        """
        Converts numpy frame (BGR) to mediapipe image object (SRGB).
        Mediapipe can only handle this image format!

        Returns
        -------
        frame : mp.Image
            current frame in Mediapipe SRGB Image format

        INFO
        ----
        If no pre-processing has been applied, the original frame is returned.
        Otherwise the pre-processed frame is returned.
        """
        if not len(self.__data):
            return
        rtrn = self.__pre_processed
        if not self.__pre_processed.any():
            rtrn = self.__data
        return mp.Image(image_format=mp.ImageFormat.SRGB, data=rtrn)

    def to_rgb(self) -> np.ndarray:
        """
        Converts mp.Image to RGB image.

        Returns
        -------
        frame : np.ndarray
            current frame in rgb color format, shape
            (height, width, channels)
        """
        return cv2.cvtColor(self.__data, cv2.COLOR_BGR2RGB)

    def to_grayscale(self) -> np.ndarray:
        """'
        returns a copy of the current frame in grayscale.
        """
        return cv2.cvtColor(self.__data, cv2.COLOR_BGR2GRAY)

    def sharp_motion(self, prev_frame: np.ndarray):
        """
        Tries to sharpen the motion in the current frame by sharpen the
        difference between two consecutive frames.
        """
        frame_diff = cv2.absdiff(self.__data, prev_frame)
        motion_mask = cv2.cvtColor(frame_diff, cv2.COLOR_BGR2GRAY)
        _, motion_mask = cv2.threshold(motion_mask, 120, 255, cv2.THRESH_BINARY)
        sharpened_motion = cv2.filter2D(
            self.__data, -1, np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        )
        sharpened_frame = cv2.bitwise_and(self.__data, self.__data, mask=motion_mask)
        sharpened_frame += cv2.bitwise_and(
            sharpened_motion, sharpened_motion, mask=~motion_mask
        )
        self.__pre_processed = sharpened_frame

    def bilateral_filter(self, output_result=False):
        """
        Applies a bilateral smoothing filter to the currentframe.
        One spacial gaussian (distance between pixels) and one range
        gaussian (intensity similarity) is used.

        The output is by default saved in an internal buffer.

        Parameters
        ----------
        output_result : bool
            if False (default), the filtered output will just be used
            internally.
            if True, the filtered output will also be saved as current frame.
        """
        cv2.bilateralFilter(
            src=self.__data,
            d=15,
            sigmaColor=75,
            sigmaSpace=75,
            dst=self.__pre_processed,
        )
        if output_result:
            self.__data = np.copy(self.__pre_processed)
            self.__pre_processed = np.empty_like(self.__data)

    def apply_filter(self, kernel=[], inplace=False):
        """
        applies a 2D filter via convolution to the current frame.

        Parameters
        ----------
        kernel : array-like
            2D filter kernel used for convolution
        inplace : bool
            if True, the convolution is performed directly on the current frame.
            if False, a copy is created on which the convolution is performed.

        Returns
        -------
        frame : np.ndarray
            filtered frame. ONLY IF inplace is false.
        """
        if len(kernel) > 0:
            kernel = np.array(kernel)
            if len(kernel.shape) != 2:
                print(
                    "kernel of 2D shape expected, got {}D shape".format(
                        len(kernel.shape)
                    )
                )
        else:
            kernel = np.array([[0, 0, 0, 0, 1, 0, 0, 0, 0]], dtype="f4")
        if inplace:
            cv2.filter2D(src=self.__data, ddepth=-1, kernel=kernel, dst=self.__data)
        else:
            cv2.filter2D(
                src=self.__data, ddepth=-1, kernel=kernel, dst=self.__pre_processed
            )

    def stabilize(self, prev_frame: np.ndarray):
        """
        Tries to stabilze the current frame by two conecutive frames.

        CAUTION
        --------
        This function is very costly.
        Thus, if used as pre-processing stage, it drastically influence the
        overall performance.
        """
        current_grayscale = self.to_grayscale()
        flow = cv2.calcOpticalFlowFarneback(
            prev_frame, current_grayscale, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        dx = flow[:, :, 0].mean()
        dy = flow[:, :, 1].mean()
        translation_matrix = np.float32([[1, 0, -dx], [0, 1, -dy]])
        cv2.warpAffine(
            self.__data,
            translation_matrix,
            (self.__data.shape[1], self.__data.shape[0]),
            dst=self.__pre_processed,
        )

    def data(self) -> np.ndarray:
        """
        Returns the frame in numpy format.

        Returns
        -------
        frame : np.ndarray
            current frame data, shape (height, width, channels).
        """
        return self.__data

    def knee_angles(self) -> np.ndarray:
        """
        Returns
        -------
        knee_angles : np.ndarray
            knee_angles[0]: right knee angle, knee_angles[1]: left knee angle.
        """
        return np.array([self.__right_knee_angle, self.__left_knee_angle], dtype="f4")

    def foot_pos(self):
        """
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
        """
        return np.array(
            [
                [self.foot_positions[0].x, self.foot_positions[1].x],
                [self.foot_positions[0].y, self.foot_positions[1].y],
            ]
        )

    def centroid_height(self) -> float:
        """
        Returns
        -------
        centroid_height : float
            normalized height of body centroid in current frame.
        """
        return self.__hip_position[1]

    def hip_pos(self) -> np.ndarray:
        """
        Returns
        -------
        hip_position : np.ndarray
            normalized hip position in current frame.
            [hip.x, hip.y]
        """
        return self.__hip_position
