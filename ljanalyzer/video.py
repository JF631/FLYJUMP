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

from utils.exception import FileNotFoundException, GeneralException
from utils.controlsignals import SharedBool
from utils.filehandler import ParameterFile
from .framebuffer import FrameBuffer
from .frame import Frame
from .posedetector import PoseDetector
from .eval import Input, EvalType

class VideoSignals(QObject):
    '''
    Defines PyQt Signals that can be emmitted by a video object.
    
    Connect to these signals to handle following events.

    Signals
    -------
    finished : pyqtSignal
        indicates that the video analysis for this object has finished
        (also emitted when process is terminated!)
    error : pyqtSignal
        emmitted when an error occured during the analysis process
    progress : pyqtSignal(int)
        publishes the current progress for the video object to keep track of 
        analysis progress in range [0 - 100] % (e.g. for progressbars)  

    '''
    finished = pyqtSignal()
    error = pyqtSignal()
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


    Parameters
    ----------
    path : str
        path to video file.
    '''
    def __init__(self, path:str, abort: SharedBool) -> None:
        super().__init__()
        self.__frame_count = 0
        self.__frame_rate = 0
        self.dims = (0, 0, 0)
        self.__open(path)
        self.__path = path
        self.__output_path = ''
        self.__detector = PoseDetector(Input.VIDEO, EvalType.REALTIME)
        self.__frame_buffer = FrameBuffer(self.__frame_count, self.dims,
                                          maxsize=2048, lock=True)
        self.__video_completed = threading.Event()
        self.abort = abort
        self.signals = VideoSignals()

    def terminate(self)->None:
        '''
        Stop running analysis.
        '''
        print("trying to abort analysis")
        self.signals.finished.emit()
    
    def __ground_contact(self, prev_foot_pos: np.ndarray, 
                         curr_foot_pos:np.ndarray):
        frames_to_consider = 2
        diff = curr_foot_pos - prev_foot_pos
        diff /= (frames_to_consider / self.__frame_rate)
        vel = np.linalg.norm(diff, axis=0)
        return np.any(vel < 0.1)

    def __open(self, path:str):
        '''
        Tries to open the video file, reads metadata and displays the first frame

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
        valid, _ = cap.read()
        if not valid:
            self.__video_completed.set()
            raise GeneralException(f"""OpenCV could not read from video file
                                   {path}""")
        # cv2.imshow("Video", first_frame)
        cap.release()

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
            # cv2.imshow("Video", frame)
            if (cv2.waitKey(0) &
                    0xFF == ord('q')):
                break
        cap.release()
        print("released input")

    def __perform_pose_detection(self):
        '''
        Runs pose detection on frames from the frame buffer and writes the
        result to a video file.
        '''
        self.__output_path = os.path.dirname(__file__)
        self.__output_path = os.path.join(
            self.__output_path,
            f'../output/{self.get_filename()}_analyzed.mp4'
        )
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(self.__output_path, fourcc, 30, (self.dims[1],
                                                               self.dims[0]))
        playback = False
        lost_frames = 0
        velocity_frames = 2
        foot_pos = np.empty((2,2), dtype='f4')
        foot_pos2 = np.empty((2,2), dtype='f4')
        counter = 0
        frame = Frame()
        param_file = ParameterFile(self.signals, self.get_filename())
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
                res = self.__detector.get_body_key_points(
                    frame.to_mediapipe_image(), counter)
                if res.pose_landmarks:
                    frame.annotate(res.pose_landmarks, as_overlay=True)
                    param_file.save(frame)
                    if counter == 0:
                        foot_pos = frame.foot_pos()
                    if (counter % velocity_frames) == 0:
                        foot_pos2 = frame.foot_pos()
                        if self.__ground_contact(foot_pos, foot_pos2):
                            cv2.putText(frame.data(), "GROUND_CONTACT", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        foot_pos = foot_pos2
                    end = time.time()
                    fps = 1 / (end - start)
                    cv2.putText(frame.data(), f'FPS: {fps:.2f} frame: {counter}',
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                                (0, 0, 255), 2)
                    frame_params = np.hstack((
                        1 - foot_pos[1], #foot height
                        1 - frame.hip_pos().reshape(-1, 1)[1], # hip height
                        frame.knee_angles())
                    ).reshape(1, -1) #expected to have at least one row
                    # print(frame_params)
                    self.signals.update_frame_parameters.emit(frame_params)
                    self.signals.update_frame.emit(frame)
                    out.write(frame.data())
                # cv2.imshow("TEST", frame)
                counter += 1
                self.update_progress(int((counter / self.__frame_count) * 100))
                if cv2.waitKey(0) & 0xFF == ord('q'):
                    break
            elif playback and self.__video_completed.is_set():
                break
        out.release()
        frame.clear()
        lost_frames = (1 - (counter / self.__frame_count)) * 100
        print(f"""video {self.__path} analyzed \n
              lost frames: {self.__frame_count - counter} 
              ({lost_frames:.2f}%)""")
        self.signals.finished.emit()
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
        '''
        return self.__path

    def get_output_path(self)->str:
        '''
        returns output video path.

        Returns
        -------
        path : str
            output video path.
        '''
        return self.__output_path
