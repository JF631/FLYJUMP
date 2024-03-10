"""
Module for analyzing body pose in a given video.

Author: Jakob Faust (software_jaf@mx442.de)
Date: 2023-10-17
"""

import os
import threading
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PyQt5.QtCore import QObject, QRunnable, pyqtSignal, pyqtSlot

from utils.controlsignals import ControlSignals, SharedBool
from utils.exception import FileNotFoundException, GeneralException
from utils.filehandler import FileHandler, ParameterFile

from .eval import EvalType, Filter, Input
from .frame import Frame
from .framebuffer import FrameBuffer
from .posedetector import PoseDetector


class VideoSignals(QObject):
    """
    Defines PyQt Signals that can be emmitted by a video object.

    Connect to these signals to handle following events.

    Signals
    -------
    finished : pyqtSignal
        indicates that the video analysis for this object has finished
        (also emitted when process is terminated!).
    error : pyqtSignal
        emmitted when an error occured during the analysis process.
    progress : pyqtSignal(int)
        publishes the current progress for the video object to keep track of
        analysis progress in range [0 - 100] % (e.g. for progressbars).
    update_frame : pyqtSignal(Frame)
        publishes current frame.
        Currently it is used to display the frame during the analysis process.
    update_frame_parameters(np.ndarray)
        publishes all analyzed data for the current frame.
        This is currently be used for visualization.
        Each column represents data that should be added to a plot
        for one graph.
        Example:
        [
            [a11, a12, a13],
            [a21, a22, a23],
            [a31, a32, a33]
        ]
              ^    ^    ^
        In the example above [a11, a21, a31] will be added to the first plot,
        first graph.
    """

    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    progress = pyqtSignal(int)
    update_frame = pyqtSignal(Frame)
    update_frame_parameters = pyqtSignal(np.ndarray)


class Video(QRunnable):
    """
    Abstraction of a video file.
    Main purpose is to provide a convenient way to analyze a video regarding
    pose detection.

    The analysis is performed within two threads:
        1) Read video frame by frame from file, visualize the frame and add it
        to a buffer.

        2) Read from the same buffer and perform pose detection.

    Furthermore, videos can simply be played / rewind / forwarded.
    The visualization is NOT part of this class.
    In order to visualize video frames you need to connect to this class'
    update_frame(Frame) signal and visualize the emitted frames on your own.

    The analysis results are published via the
    update_frame_parameters(np.ndarray) signal.
    For detailed explanation see above.

    Parameters
    ----------
    path : str
        path to video file.
    abort : SharedBool
        whenever this bool turns True, all running threads are tried to be
        interrupted.
        Implementation is based based on C++'s std::atomic<bool>.


    """

    def __init__(self, path: str, abort: SharedBool) -> None:
        super().__init__()
        self.signals = VideoSignals()
        self.control_signals = None
        self.__frame_count = 0
        self.__frame_rate = 0
        self.dims = (0, 0, 0)
        self.__stop_flag = False
        self.__current_frame = Frame()
        self.__cap: cv2.VideoCapture = None
        self.__open(path)
        self.__path = path
        self.__output_path = path
        self.__detector = PoseDetector(Input.VIDEO, EvalType.REALTIME)
        self.__marker_overlay = True
        self.__save_filter_output = False
        self.__frame_buffer = FrameBuffer(
            self.__frame_count, self.dims, maxsize=2048, lock=True
        )
        self.__video_completed = threading.Event()
        self.abort = abort
        self.__filter: Filter = None
        self.__playback = False

    def terminate(self) -> None:
        """
        Stop running analysis.
        """
        print(f"trying to abort analysis: {self.get_output_path()}")
        self.signals.error.emit(self.get_output_path())
    
    def set_analysis_overlay(self, as_overlay: bool):
        '''
        switch analysis mode between overlay and not overlay.
        If false, body key points are shown on black background
        '''
        if self.__marker_overlay == as_overlay:
            return
        self.__marker_overlay = as_overlay

    def set_filter_output(self, output: bool):
        '''
        switch analysis mode between saving filter output.
        If false, the filtered output is not saved.
        '''
        if self.__save_filter_output == output:
            return
        print(f"filter output mode changed to: {output}")
        self.__save_filter_output = output

    def export_frame(self, path: str):
        '''
        Export current frame as image.
        Image is saved as path

        Parameters
        ----------
        path : str
            path to save the exported frame
        '''
        cv2.imwrite(path, self.__current_frame.data())

    def __ground_contact(self, prev_foot_pos: np.ndarray,
                         curr_foot_pos: np.ndarray):
        """
        Tries to detect if a foot is on ground.
        As it uses velocity as main parameter for detection, at least two foot
        positions are needed.

        Parameters
        ----------
        prev_foot_pos : np.ndarray
            first foot pos
        curr_foot_pos : np.ndarray
            second foot pos
        """
        frames_to_consider = 2
        diff = curr_foot_pos - prev_foot_pos
        diff /= frames_to_consider / self.__frame_rate
        vel = np.linalg.norm(diff, axis=0)
        return np.any(vel < 0.1)

    def __open(self, path: str):
        """
        Tries to open the video file, reads metadata and displays the first
        frame

        Parameters
        ----------
        path : str
            path to video file.
        """
        self.__cap = cv2.VideoCapture(path)
        if not self.__cap.isOpened():
            raise FileNotFoundException(
                f"""file could not be opened - check
                                        file path {path}"""
            )
        self.__frame_count = int(self.__cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.__frame_rate = self.__cap.get(cv2.CAP_PROP_FPS)
        height = int(self.__cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(self.__cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.dims = (height, width, 3)
        valid, data = self.__cap.read()
        if not valid:
            self.__video_completed.set()
            raise GeneralException(
                f"""OpenCV could not read from video file
                                   {path}"""
            )
        self.__current_frame.update(data)
        self.signals.update_frame.emit(self.__current_frame)
        # self.__cap.release()

    def play(self, frame=None):
        """
        starts video playback for the current video.

        Parameters
        ----------
        frame : int | None
            if a number is given, the video will jump to this frame number before playback.
            Otherwise, it will just start from the beginning.
        """
        self.__cap = cv2.VideoCapture(self.__path)
        self.__playback = True
        param_file = ParameterFile(self.get_analysis_path())
        param_file.load()
        takeoff_parm = param_file.get_takeoff_angle()
        takeoff_vec = None
        if takeoff_parm is not None:
            takeoff_vec = np.array([takeoff_parm[1], takeoff_parm[2]])
        if frame:
            self.__playback = False
            self.jump_to_frame(frame)
            if takeoff_vec is not None:
                dir_sign = np.sign(takeoff_vec[0] - takeoff_vec[1])
                self.show_hip_vector(frame, takeoff_vec)
                horizontal_vec = dir_sign * np.array([0.1, 0])
                self.show_hip_vector(frame, horizontal_vec)
        while self.__cap.isOpened():
            if self.abort.get() or self.__stop_flag:
                self.terminate()
                break
            if self.__playback:
                valid, frame = self.__cap.read()
                if not valid:
                    break
                self.__current_frame.update(frame)
                self.signals.update_frame.emit(self.__current_frame)
            if cv2.waitKey(int(self.__frame_rate)) & 0xFF == ord("q"):
                self.terminate()
                break
        self.__cap.release()
        cv2.destroyAllWindows()

    def rewind(self):
        """
        rewinds currently playing video by one frame.
        """
        self.pause()
        if not self.__cap:
            return
        current_frame = self.__cap.get(cv2.CAP_PROP_POS_FRAMES)
        current_frame -= 1
        if self.control_signals:
            self.control_signals.jump_to_frame.emit(int(current_frame - 1))
        else:
            self.jump_to_frame(current_frame - 1)

    def forward(self):
        """
        forwards currently playing video by one frame.
        """
        self.pause()
        if not self.__cap:
            return
        current_frame = self.__cap.get(cv2.CAP_PROP_POS_FRAMES)
        current_frame += 1
        if self.control_signals:
            self.control_signals.jump_to_frame.emit(int(current_frame))
        else:
            self.jump_to_frame(current_frame)

    def toggle(self):
        """
        plays / pauses video playback.
        """
        self.__playback = not self.__playback

    def pause(self):
        """
        pauses video playback.
        """
        self.__playback = False

    def stop(self):
        """
        stops video playback.
        """
        print("stop invoked")
        self.__cap.release()
        self.__stop_flag = True

    def jump_to_frame(self, frame):
        """
        Jumps to a certain frame in video playback.

        Parameters
        ----------
        frame : int
            frame number to which the player should jump.
        cap : cv2.VideoCapture
            openCV video capture object.
            if None is provided, a new one for the current video object will
            be created
        """
        if frame > self.__frame_count or frame < 0:
            print('frame index too large')
            return
        if not self.__cap:
            print('nothing to play')
            return
        self.pause()
        self.__cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        valid, frame = self.__cap.read()
        if valid:
            self.__current_frame.update(frame)
            self.signals.update_frame.emit(self.__current_frame)

    def __read_video_file(self):
        """
        reads video frame by frame from file, visualizes the frame and adds it
        to a frame buffer.
        """
        cap = cv2.VideoCapture(self.__path)
        while cap.isOpened():
            if self.abort.get():
                self.terminate()
                return
            valid, frame = cap.read()
            if not valid:
                self.__video_completed.set()
                print("video ended or an error occurred")
                break
            self.__frame_buffer.add(frame)
            if cv2.waitKey(0) & 0xFF == ord("q"):
                break
        cap.release()
        print("released input")

    def set_filter(self, filter: Filter):
        '''
        set filter to use during video analysis
        '''
        self.__filter = filter
    
    def set_eval_type(self, eval_type: EvalType):
        '''
        set Evaluation type to use during video analysis
        '''
        if eval_type.name == self.__detector.get_eval_type():
            return
        self.__detector = PoseDetector(Input.VIDEO, eval_type)

    def __perform_pose_detection(self):
        """
        Runs pose detection on frames from the frame buffer and writes the
        result to a video file.
        """
        self.__output_path = FileHandler.create_current_folder()
        self.__output_path = os.path.join(
            self.__output_path, f"{self.get_filename()}_analyzed.mp4"
        )
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter(
            self.__output_path, fourcc, 30, (self.dims[1], self.dims[0])
        )
        playback = False
        hip_height = []
        knee_angles = []
        lost_frames = 0
        velocity_frames = 2
        foot_pos = np.empty((2, 2), dtype="f4")
        foot_pos2 = np.empty((2, 2), dtype="f4")
        counter = 0
        analyzed_counter = 0
        self.__current_frame.update(self.__frame_buffer.pop())
        param_file = ParameterFile(self.get_analysis_path(), self.signals)
        while True:
            if self.abort.get():
                self.terminate()
                break
            if self.__frame_buffer.current_frames() > 0:
                start = time.time()
                self.__current_frame.update(self.__frame_buffer.pop())
                playback = True
                if self.__current_frame is None:
                    print("Empty frame received")
                    break
                self.__current_frame.pre_process(
                    self.__filter, inplace=self.__save_filter_output
                )
                res = self.__detector.get_body_key_points(
                    self.__current_frame.to_mediapipe_image(), counter
                )
                if res.pose_landmarks:
                    self.__current_frame.annotate(
                        res.pose_landmarks, as_overlay=self.__marker_overlay
                    )
                    param_file.save(self.__current_frame)
                    if counter == 0:
                        foot_pos = self.__current_frame.foot_pos()
                        hip_pos = self.__current_frame.hip_pos()
                        if (
                            np.any(foot_pos > 1.0)
                            or np.any(foot_pos < 0.0)
                            or np.any(hip_pos > 1.0)
                            or np.any(hip_pos < 0.0)
                        ):
                            continue
                    if (counter % velocity_frames) == 0:
                        foot_pos2 = self.__current_frame.foot_pos()
                        # if self.__ground_contact(foot_pos, foot_pos2):
                        #     cv2.putText(
                        #         self.__current_frame.data(),
                        #         "GROUND_CONTACT",
                        #         (10, 120),
                        #         cv2.FONT_HERSHEY_SIMPLEX,
                        #         1,
                        #         (0, 0, 255),
                        #         2,
                        #     )
                        foot_pos = foot_pos2
                    end = time.time()
                    fps = 1 / (end - start)
                    # cv2.putText(
                    #     self.__current_frame.data(),
                    #     f"FPS: {fps:.2f} frame: {analyzed_counter}",
                    #     (10, 60),
                    #     cv2.FONT_HERSHEY_SIMPLEX,
                    #     1,
                    #     (0, 0, 255),
                    #     2,
                    # )
                    hip_height.append(1 - self.__current_frame.hip_pos()[1])
                    knee_angles.append(self.__current_frame.knee_angles())
                    frame_params = np.hstack(
                        (
                            1 - foot_pos[1],  # foot height
                            1 - self.__current_frame.hip_pos().reshape(-1, 1)[1],  # hip height
                            self.__current_frame.knee_angles(),
                        )
                    ).reshape(
                        1, -1
                    )  # expected to have at least one row
                    self.signals.update_frame_parameters.emit(frame_params)
                    self.signals.update_frame.emit(self.__current_frame)
                    out.write(self.__current_frame.data())
                    analyzed_counter += 1
                counter += 1
                self.update_progress(int((counter / self.__frame_count) * 100))
                if cv2.waitKey(0) & 0xFF == ord("q"):
                    break
            elif playback and self.__video_completed.is_set():
                break
        out.release()
        # self.__current_frame.clear()s
        lost_frames = (1 - (counter / self.__frame_count)) * 100
        param_file.close()
        tkf_frame = self.takeoff_frame(hip_height=hip_height,
                                       knee_angles=knee_angles, full=False)
        if tkf_frame:
            print(f"takeoff detected at {tkf_frame}")
            param_file.add_metadata(('takeoff', tkf_frame))
            tkf_angle = self.takeoff_angle(takeoff_index=tkf_frame)
            if tkf_angle:
                param_file.add_metadata(('takeoff_angle', tkf_angle))
                print(f'takeoff angle saved: {tkf_angle}')
        print(
            f"""video {self.__path} analyzed \n
              lost frames: {self.__frame_count - counter} 
              ({lost_frames:.2f}%)"""
        )
        self.signals.finished.emit(self.get_analysis_path())
        cv2.destroyAllWindows()

    def update_progress(self, current_progress: int):
        """
        Emits progress signal with current analysis progress in percent
        """
        self.signals.progress.emit(current_progress)

    def run(self) -> None:
        """
        Starts the body pose analyzing process on two threads.
        One thread reads in video from storage and loads it into a frame
        buffer, while the other one performs pose detection using frames
        from this buffer.
        """
        load_thread = threading.Thread(target=self.__read_video_file)
        visualize_thread = threading.Thread(target=self.__perform_pose_detection)
        load_thread.start()
        visualize_thread.start()
        load_thread.join()
        visualize_thread.join()

    def get_filename(self) -> str:
        """
        Filename without path and extension.

        Returns
        -------
        filename : str
            filename without extension and path. (so e.g. just 'vid0')
        """
        return os.path.splitext(os.path.basename(self.__path))[0]

    def get_path(self) -> str:
        """
        returns input video path.

        Returns
        -------
        path : str
            input video path.
            (e.g. 'C:/Downloads/vid0.mp4')
        """
        return self.__path

    def get_base_path(self) -> str:
        """
        returns input video path without file extension.

        Returns
        -------
        path : str
            input video path without file extension.
            (e.g. 'C:/FLYJUMP/analysis/2023-01-01/vid0')
        """
        return os.path.splitext(self.__output_path)[0]

    def get_analysis_path(self) -> str:
        """
        returns output analysis file path.

        Returns
        -------
        path : str
            output analysis file path
            (e.g. 'C:/FLYJUMP/analysis/2023-01-01/vid0.hdf5').
        """
        return self.get_base_path() + ".hdf5"

    def get_output_path(self) -> str:
        """
        returns output video path.

        Returns
        -------
        path : str
            output video path.
            (e.g. 'C:/FLYJUMP/analysis/2023-01-01/vid0.mp4')
        """
        return self.__output_path
    
    def show_hip_vector(self, frame_index: int, vector: np.ndarray):
        '''
        Visualize vector with origin on CM position.
        The annotation is NOT saved persistantly.

        Parameters
        ----------
        frame_index : int
            frame to draw the vector on
        vector : np.ndarray
            directional vector.
            [x-dir, y-dir].
        '''
        analysis_path = self.get_analysis_path()
        param_file = ParameterFile(analysis_path)
        param_file.load()
        start_point = 1 - param_file.get_hip_pos(index=frame_index)
        width_height = self.dims[:2][::-1]
        sp = np.int32(start_point * width_height)
        ep = np.int32((start_point + vector) * width_height)
        cv2.arrowedLine(self.__current_frame.data(), sp, ep, (0, 255, 0), 2)
        self.__current_frame.update(self.__current_frame.data())
        self.signals.update_frame.emit(self.__current_frame)

    
    def takeoff_angle(self, hip_pos: np.ndarray = None,
                      takeoff_index: int = None):
        '''
        Calculates takeoff angle to a given takeoff frame index.
        Takeoff angle is defined as angle between horizontal CM vector
        the velocity vector of the CM.
        The velocity vector is calculated over frame_rate / 10 frames.

        Parameters
        ----------
        hip_pos : np.ndarray
            hip position array of the respective x and y coordinates holding
            the hip positions for each frame
            if None it is automatically retrieved from the parameter file.
        takeoff_index : int
            index of the takeoff frame.
            if None, it is automatically retrieved from the parameter file.
        
        Returns
        -------
        takeoff_angle : float
            calculated takeoff angle.
            if something went wrong, None is returned (e.g. no takeoff frame
            was found in parameter file)
        '''
        analysis_path = self.get_analysis_path()
        if not os.path.exists(analysis_path):
            print(f"file not found: {analysis_path}!")
            self.signals.error.emit(self.get_output_path())
            return None
        param_file = ParameterFile(analysis_path)
        if hip_pos is None:
            param_file.load()
            hip_pos = param_file.get_hip_pos()
        if not takeoff_index:
            takeoff_index = param_file.get_takeoff_frame()
            if not takeoff_index:
                return None
        self.jump_to_frame(takeoff_index[0])
        frame_offset = int(self.__frame_rate / 10)
        if takeoff_index[0] + frame_offset > (len(hip_pos) - 1):
            return None
        takeoff_pos = 1 - hip_pos[takeoff_index[0], :]
        offset_pos = 1 -  hip_pos[takeoff_index[0] + frame_offset, :]
        dir_sign = np.sign(offset_pos[0] - takeoff_pos[0])
        hip_vel_vec = (-1 / frame_offset) * (takeoff_pos - offset_pos)
        hip_horizontal_vec = dir_sign * np.array([1, 0])
        hip_vel_vec *= 20
        hip_horizontal_vec *= 20
        hip_vel_vec_norm = np.linalg.norm(hip_vel_vec)
        hip_horizontal_vec_norm = np.linalg.norm(hip_horizontal_vec)
        if hip_vel_vec_norm == 0 or hip_horizontal_vec_norm == 0:
            return None
        rtrn = np.rad2deg(
            np.arccos((np.dot(hip_vel_vec, hip_horizontal_vec)) / 
                      (hip_vel_vec_norm * hip_horizontal_vec_norm)))
        return (rtrn, *hip_vel_vec)

    def takeoff_frame(self, hip_height: np.ndarray = None,
                      knee_angles: np.ndarray = None, full: bool = False):
        """
        detects the frame that shows the takeoff image.
        It uses the hip position, actually the hip height, to detect the takeoff.

        Parameters
        -----------
        hip_height : np.ndarray
            flat array holding the hip height over time.
            (one value per frame).
            If no array is given, the analysis must have already been
            performed, so that an according .hdf5 file is present.
            Then, the values can be extracted automatically from the analysis
            file.
        full : bool
            also detect other frame candidates.
            in error range of + or - 1% from the detected frame.

        Returns
        --------
        if full = True: (frame, candidates[]) : (int, np.ndarray)
            frame - that is most likely to be the takeoff frame,
            candidates - holds all other frames that might be reasonable.
        if full = False: frame : int
            detected takeoff frame that is most likely

        if no frame is detected or an error occured: None is returned
        """
        self.__open(self.__path)
        analysis_path = self.get_analysis_path()
        if not os.path.exists(analysis_path):
            print(f"file not found: {analysis_path}!")
            self.signals.error.emit(self.get_output_path())
            return None
        param_file = ParameterFile(analysis_path)
        if hip_height is None:
            param_file.load()
            hip_height = param_file.get_hip_height()
        if knee_angles is None:
            param_file.load()
            knee_angles = param_file.get_knee_angles()
        print(knee_angles)
        total_error = 100
        changing_points = (0, 0)
        if full:
            possible_indices = []
        runup_coeffs = []
        jump_coeffs = []
        land_coeffs = []
        n = len(hip_height)
        if full:
            all_errors = np.empty((n, n), dtype="f4")
        for i in range(2, n - 4):
            for j in range(i + 2, n - 2):
                x_runup = np.arange(i)  # hip_height[:i]
                x_jump = np.arange(j - i)  # hip_height[i:j]
                x_landing = np.arange(n - j)  # hip_height[j:]
                hip_fit_runup, residuals_runup, _, _, _ = np.polyfit(
                    x_runup, hip_height[:i], 1, full=True
                )
                hip_fit_jump, residuals_jump, _, _, _ = np.polyfit(
                    x_jump, hip_height[i:j], 2, full=True
                )
                hip_fit_landing, residuals_landing, _, _, _ = np.polyfit(
                    x_landing, hip_height[j:], 1, full=True
                )
                fitting_error = residuals_runup + residuals_jump + residuals_landing
                if fitting_error and full:
                    all_errors[i, j] = fitting_error
                """
                since we already know the fitted jumping curve must be of form
                -ax^2 + bx + c, we know a = hip_fit_jump[0] < 0.
                And, as the jumping leg is fully extended during takeoff, the
                matching knee angle must be above 170 degrees
                """
                if (fitting_error < total_error and hip_fit_jump[0] < 0
                    and np.any(knee_angles[i] >= 170.0)):
                    total_error = fitting_error
                    changing_points = (i, j)
                    runup_coeffs = hip_fit_runup
                    jump_coeffs = hip_fit_jump
                    land_coeffs = hip_fit_landing
        if full:
            lower_bound = total_error - total_error * 0.1
            upper_bound = total_error + total_error * 0.1
            print("total error: {}".format(total_error))
            possible_indices = np.argwhere(
                (all_errors >= lower_bound) & (all_errors <= upper_bound)
            )
            print("shape: {}".format(possible_indices.shape))
            print("possible: {}".format(possible_indices))
        hip_runup = np.poly1d(runup_coeffs)
        hip_jump = np.poly1d(jump_coeffs)
        x_runup = np.arange(0, changing_points[0])
        x_jump = np.arange(changing_points[0], changing_points[1])
        x_landing = np.arange(changing_points[1], n)
        hip_runup = np.poly1d(runup_coeffs)
        hip_jump = np.poly1d(jump_coeffs)
        hip_landing = np.poly1d(land_coeffs)
        plt.xlabel("t [frames]")
        plt.ylabel("height [norm. pix]")
        plt.scatter(
            changing_points[0], hip_height[changing_points[0]], label="takeoff", c="red"
        )
        plt.plot(hip_height, label="hip")
        plt.plot(x_runup, hip_runup(x_runup), label="runup")
        plt.plot(x_jump, hip_jump(np.arange(len(x_jump))), label="jump")
        plt.plot(x_landing, hip_landing(np.arange(len(x_landing))), label="landing")
        plt.legend()
        file_name = "test.png"
        plt.savefig(file_name)

        if full:
            return (changing_points, possible_indices)
        return changing_points

    def set_control_signals(self, control_signals: ControlSignals):
        """'
        sets program control signals to current video.

        Parameters
        ----------
        control_signals : ControlSignals
            programm control signals
        """
        self.control_signals = control_signals
        self.control_signals.jump_to_frame.connect(self.jump_to_frame)
        self.control_signals.export_frame.connect(self.export_frame)
        self.control_signals.change_filter_output_mode.connect(
            self.set_filter_output)
