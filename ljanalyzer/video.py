'''
Module for analyzing body pose in a given video.

Author: Jakob Faust (software_jaf@mx442.de)
Date: 2023-10-17
'''

import threading
import os
import time

import cv2
from PyQt5.QtCore import QRunnable, QObject, pyqtSignal
import numpy as np
import matplotlib.pyplot as plt

from utils.exception import FileNotFoundException, GeneralException
from utils.controlsignals import SharedBool, ControlSignals
from utils.filehandler import ParameterFile, FileHandler
from .framebuffer import FrameBuffer
from .frame import Frame
from .posedetector import PoseDetector
from .eval import Input, EvalType, Filter

class VideoSignals(QObject):
    '''
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
    '''
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    progress = pyqtSignal(int)
    update_frame = pyqtSignal(Frame)
    update_frame_parameters = pyqtSignal(np.ndarray)

class Video(QRunnable):
    '''
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
        

    '''
    def __init__(self, path:str, abort: SharedBool) -> None:
        super().__init__()
        self.signals = VideoSignals()
        self.control_signals = None
        self.__frame_count = 0
        self.__frame_rate = 0
        self.dims = (0, 0, 0)
        self.__stop_flag = False
        self.__open(path)
        self.__path = path
        self.__output_path = path
        self.__detector = PoseDetector(Input.VIDEO, EvalType.FULL)
        self.__frame_buffer = FrameBuffer(self.__frame_count, self.dims,
                                          maxsize=2048, lock=True)
        self.__video_completed = threading.Event()
        self.abort = abort
        self.__filter: Filter = None
        self.__playback = False
        self.__cap: cv2.VideoCapture = None

    def terminate(self)->None:
        '''
        Stop running analysis.
        '''
        print(f"trying to abort analysis: {self.get_output_path()}")
        self.signals.error.emit(self.get_output_path())

    def __ground_contact(self, prev_foot_pos:np.ndarray,
                         curr_foot_pos:np.ndarray):
        '''
        Tries to detect if a foot is on ground.
        As it uses velocity as main parameter for detection, at least two foot
        positions are needed.

        Parameters
        ----------
        prev_foot_pos : np.ndarray
            first foot pos
        curr_foot_pos : np.ndarray
            second foot pos
        '''
        frames_to_consider = 2
        diff = curr_foot_pos - prev_foot_pos
        diff /= (frames_to_consider / self.__frame_rate)
        vel = np.linalg.norm(diff, axis=0)
        return np.any(vel < 0.1)

    def __open(self, path:str):
        '''
        Tries to open the video file, reads metadata and displays the first
        frame

        Parameters
        ----------
        path : str
            path to video file.
        '''
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise FileNotFoundException(f"""file could not be opened - check
                                        file path {path}""")
        self.__frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.__frame_rate = cap.get(cv2.CAP_PROP_FPS)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.dims = (height, width, 3)
        valid, data = cap.read()
        if not valid:
            self.__video_completed.set()
            raise GeneralException(f"""OpenCV could not read from video file
                                   {path}""")
        frame = Frame(data)
        self.signals.update_frame.emit(frame)
        cap.release()

    def play(self, frame = None):
        '''
        starts video playback for the current video.
        
        Parameters
        ----------
        frame : int | None
            if a number is given, the video will jump to this frame number before playback.
            Otherwise, it will just start from the beginning.
        '''
        self.__cap = cv2.VideoCapture(self.__path)
        self.__playback = True
        current_frame = Frame()
        if frame:
            self.__playback = False
            self.jump_to_frame(frame)
        while self.__cap.isOpened():
            if self.abort.get() or self.__stop_flag:
                self.terminate()
                break
            if self.__playback:
                valid, frame = self.__cap.read()
                if not valid:
                    break
                current_frame.update(frame)
                self.signals.update_frame.emit(current_frame)
            if (cv2.waitKey(int(self.__frame_rate)) &
                   0xFF == ord('q')):
                self.terminate()
                break
        self.__cap.release()
        cv2.destroyAllWindows()

    def rewind(self):
        '''
        rewinds currently playing video by one frame.
        '''
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
        '''
        forwards currently playing video by one frame. 
        '''
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
        '''
        plays / pauses video playback.
        '''
        self.__playback = not self.__playback

    def pause(self):
        '''
        pauses video playback.
        '''
        self.__playback = False

    def stop(self):
        '''
        stops video playback.
        '''
        print("stop invoked")
        self.__cap.release()
        self.__frame_buffer
        self.__stop_flag = True
    
    def jump_to_frame(self, frame):
        '''
        Jumps to a certain frame in video playback.
        
        Parameters
        ----------
        frame : int
            frame number to which the player should jump.
        cap : cv2.VideoCapture
            openCV video capture object.
            if None is provided, a new one for the current video object will
            be created
        '''
        if frame > self.__frame_count or frame < 0:
            return
        if not self.__cap:
            return
        self.pause()
        self.__cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        valid, frame = self.__cap.read()
        if valid:
            current_frame = Frame(frame)
            self.signals.update_frame.emit(current_frame)

    def __read_video_file(self):
        '''
        reads video frame by frame from file, visualizes the frame and adds it
        to a frame buffer.
        '''
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
            if (cv2.waitKey(0) &
                   0xFF == ord('q')):
                break
        cap.release()
        print("released input")
    
    def set_filter(self, filter:Filter):
        self.__filter = filter

    def __perform_pose_detection(self):
        '''
        Runs pose detection on frames from the frame buffer and writes the
        result to a video file.
        '''
        self.__output_path = FileHandler.create_current_folder()
        self.__output_path = os.path.join(
            self.__output_path,
            f'{self.get_filename()}_analyzed.mp4'
        )
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(self.__output_path, fourcc, 30, (self.dims[1],
                                                               self.dims[0]))
        playback = False
        hip_height = []
        lost_frames = 0
        velocity_frames = 2
        foot_pos = np.empty((2,2), dtype='f4')
        foot_pos2 = np.empty((2,2), dtype='f4')
        counter = 0
        analyzed_counter = 0
        frame = Frame(self.__frame_buffer.pop())
        param_file = ParameterFile(self.get_analysis_path(), self.signals)
        while True:
            if self.abort.get():
                self.terminate()
                break
            if self.__frame_buffer.current_frames() > 0:
                start = time.time()
                frame.update(self.__frame_buffer.pop())
                playback = True
                if frame is None:
                    print("Empty frame received")
                    break
                frame.pre_process(self.__filter, inplace=True)
                res = self.__detector.get_body_key_points(
                    frame.to_mediapipe_image(), counter)
                if res.pose_landmarks:
                    frame.annotate(res.pose_landmarks, as_overlay=False)
                    param_file.save(frame)
                    if counter == 0:
                        foot_pos = frame.foot_pos()
                        hip_pos = frame.hip_pos()
                        if (np.any(foot_pos > 1.0) or np.any(foot_pos < 0.0)
                            or
                            np.any(hip_pos > 1.0) or np.any(hip_pos < 0.0)):
                            continue
                    if (counter % velocity_frames) == 0:
                        foot_pos2 = frame.foot_pos()
                        if self.__ground_contact(foot_pos, foot_pos2):
                            cv2.putText(frame.data(), "GROUND_CONTACT",
                                        (10, 120), cv2.FONT_HERSHEY_SIMPLEX,
                                        1, (0, 0, 255), 2)
                        foot_pos = foot_pos2
                    end = time.time()
                    fps = 1 / (end - start)
                    cv2.putText(frame.data(), f'FPS: {fps:.2f} frame: {analyzed_counter}',
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 0, 255), 2)
                    hip_height.append(1 - frame.hip_pos()[1])
                    frame_params = np.hstack((
                        1 - foot_pos[1], #foot height
                        1 - frame.hip_pos().reshape(-1, 1)[1], # hip height
                        frame.knee_angles())
                    ).reshape(1, -1) #expected to have at least one row
                    self.signals.update_frame_parameters.emit(frame_params)
                    self.signals.update_frame.emit(frame)
                    out.write(frame.data())
                    analyzed_counter += 1
                counter += 1
                self.update_progress(int((counter / self.__frame_count) * 100))
                if cv2.waitKey(0) & 0xFF == ord('q'):
                    break
            elif playback and self.__video_completed.is_set():
                break
        out.release()
        frame.clear()
        lost_frames = (1 - (counter / self.__frame_count)) * 100
        param_file.close()
        tkf_frame = self.takeoff_frame(hip_height=hip_height,
                                       full=True)[0]
        if tkf_frame:
            print(f"takeoff detected at {tkf_frame}")
            param_file.add_metadata(('takeoff', tkf_frame))
        print(f"""video {self.__path} analyzed \n
              lost frames: {self.__frame_count - counter} 
              ({lost_frames:.2f}%)""")
        self.signals.finished.emit(self.get_analysis_path())
        cv2.destroyAllWindows()

    def update_progress(self, current_progress: int):
        '''
        Emits progress signal with current analysis progress in percent 
        '''
        self.signals.progress.emit(current_progress)

    def run(self) -> None:
        '''
        Starts the body pose analyzing process on two threads.
        One thread reads in video from storage and loads it into a frame 
        buffer, while the other one performs pose detection using frames
        from this buffer.
        '''
        load_thread = threading.Thread(target=self.__read_video_file)
        visualize_thread = threading.Thread(
            target=self.__perform_pose_detection)
        load_thread.start()
        visualize_thread.start()
        load_thread.join()
        visualize_thread.join()

    def get_filename(self)->str:
        '''
        Filename without path and extension.

        Returns
        -------
        filename : str
            filename without extension and path. (so e.g. just 'vid0')
        '''
        return os.path.splitext(os.path.basename(self.__path))[0]

    def get_path(self)->str:
        '''
        returns input video path.

        Returns
        -------
        path : str
            input video path.
            (e.g. 'C:/Downloads/vid0.mp4')
        '''
        return self.__path
    
    def get_base_path(self)->str:
        '''
        returns input video path without file extension.

        Returns
        -------
        path : str
            input video path without file extension.
            (e.g. 'C:/FLYJUMP/analysis/2023-01-01/vid0')
        '''
        return os.path.splitext(self.__output_path)[0]
    
    def get_analysis_path(self)->str:
        '''
        returns output analysis file path.

        Returns
        -------
        path : str
            output analysis file path
            (e.g. 'C:/FLYJUMP/analysis/2023-01-01/vid0.hdf5').
        '''
        return self.get_base_path() + '.hdf5'

    def get_output_path(self)->str:
        '''
        returns output video path.

        Returns
        -------
        path : str
            output video path.
            (e.g. 'C:/FLYJUMP/analysis/2023-01-01/vid0.mp4')
        '''
        return self.__output_path

    def takeoff_frame(self, hip_height: np.ndarray = None, full = False):
        '''
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
        '''
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
        total_error = 100
        changing_points = (0,0)
        if full:
            possible_indices = []
        runup_coeffs = []
        jump_coeffs = []
        land_coeffs = []
        n = len(hip_height)
        if full:
            all_errors = np.empty((n, n), dtype='f4')
        for i in range(2, n - 4):
            for j in range(i + 2, n - 2):
                x_runup = np.arange(i) # hip_height[:i]
                x_jump = np.arange(j - i) # hip_height[i:j]
                x_landing = np.arange(n - j) # hip_height[j:]
                hip_fit_runup, residuals_runup, _, _, _ = np.polyfit(
                    x_runup,hip_height[:i], 1, full=True)
                hip_fit_jump, residuals_jump, _, _, _ = np.polyfit(
                    x_jump, hip_height[i:j], 2, full=True)
                hip_fit_landing, residuals_landing, _, _, _ = np.polyfit(
                    x_landing, hip_height[j:], 1, full=True)
                fitting_error = (residuals_runup + residuals_jump +
                                 residuals_landing)
                if fitting_error and full:
                    all_errors[i, j] = fitting_error
                '''
                since we already know the fitted jumping curve must be of form
                -ax^2 + bx + c, we know a = hip_fit_jump[0] < 0.
                '''
                if fitting_error < total_error and hip_fit_jump[0] < 0:
                    total_error = fitting_error
                    changing_points = (i,j)
                    runup_coeffs = hip_fit_runup
                    jump_coeffs = hip_fit_jump
                    land_coeffs = hip_fit_landing
        if full:
            lower_bound = total_error - total_error * 0.1
            upper_bound = total_error + total_error * 0.1
            print('total error: {}'.format(total_error))
            possible_indices = np.argwhere((all_errors >= lower_bound) &
                                           (all_errors <= upper_bound))
            print('shape: {}'.format(possible_indices.shape))
            print('possible: {}'.format(possible_indices))
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
        plt.scatter(changing_points[0], hip_height[changing_points[0]],
                    label='takeoff', c='red')
        plt.plot(hip_height, label='hip')
        plt.plot(x_runup, hip_runup(x_runup), label="runup")
        plt.plot(x_jump, hip_jump(np.arange(len(x_jump))), label="jump")
        plt.plot(x_landing, hip_landing(np.arange(len(x_landing))),
                 label="landing")
        plt.legend()
        file_name ='test.png'
        plt.savefig(file_name)

        if full:
            return (changing_points, possible_indices)
        return changing_points

    def set_control_signals(self, control_signals: ControlSignals):
        ''''
        sets program control signals to current video.

        Parameters
        ----------
        control_signals : ControlSignals
            programm control signals
        '''
        self.control_signals = control_signals
        self.control_signals.jump_to_frame.connect(self.jump_to_frame)
