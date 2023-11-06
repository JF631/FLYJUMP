'''
Module for analyzing body pose in a given video.

Author: Jakob Faust (software_jaf@mx442.de)
Date: 2023-10-17
'''

import threading
import os
import time
from math import sqrt

import cv2
from PyQt5.QtCore import QRunnable, QObject, pyqtSignal

from .framebuffer import FrameBuffer
from .frame import Frame
from .posedetector import PoseDetector
from .eval import Input, EvalType
from utils.exception import FileNotFoundException, GeneralException
from utils.controlsignals import SharedBool

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
                                          maxsize=1024, lock=True)
        self.__video_completed = threading.Event()
        self.abort = abort
        self.signals = VideoSignals()
    
    def terminate(self)->None:
        '''
        Stop running analysis.
        '''
        print("trying to abort analysis")
        self.signals.finished.emit()

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
        counter = 0
        foot_x1 = 0
        foot_x2 = 0
        foot_y1 = 0
        foot_y2 = 0
        vel_x = 0.0
        vel_y = 0.0
        while True:
            if self.abort.get():
                self.terminate()
                break
            if self.__frame_buffer.current_frames() > 0:
                start = time.time()
                frame = Frame(self.__frame_buffer.pop())
                playback = True
                if frame is None:
                    print("Empty frame received")
                    break
                mp_image = frame.to_mediapipe_image()
                res = self.__detector.get_body_key_points(mp_image, counter)
                if res.pose_landmarks:
                    frame.annotate(res.pose_landmarks, as_overlay=False)
                    if counter == 0:
                        foot_y1 = frame.foot_pos().y
                        foot_x1 = frame.foot_pos().x
                    if (counter % velocity_frames) == 0:
                        foot_y2 = frame.foot_pos().y
                        foot_x2 = frame.foot_pos().x
                        vec_x = foot_x2 - foot_x1
                        foot_x1 = foot_x2
                        vel_x = vec_x / (velocity_frames / self.__frame_rate)
                        vec_y = foot_y2 - foot_y1
                        foot_y1 = foot_y2
                        vel_y = vec_y / (velocity_frames / self.__frame_rate)
                        if sqrt(vel_x**2 + vel_y**2) < 0.1:
                            cv2.putText(frame.data(), f'GROUND_CONTACT', (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.putText(frame.data(), f'velx: {vel_x:.4f} vely: {vel_y: .4f}', (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    end = time.time()
                    fps = 1 / (end - start)
                    cv2.putText(frame.data(), f'FPS: {fps:.2f}', (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    out.write(frame.data())
                # cv2.imshow("TEST", frame)
                counter += 1
                self.update_progress(int((counter / self.__frame_count) * 100))
                if cv2.waitKey(0) & 0xFF == ord('q'):
                    break
            elif playback and self.__video_completed.is_set():
                break
        out.release()
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
