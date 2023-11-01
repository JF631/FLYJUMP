'''
Module for analyzing body pose in a given video.

Author: Jakob Faust (software_jaf@mx442.de)
Date: 2023-10-17
'''

import threading
import os
import time

import cv2

from .framebuffer import FrameBuffer
from .frame import Frame
from .posedetector import PoseDetector
from .eval import Input, EvalType
from utils.exception import FileNotFoundException, GeneralException

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
        self.__frame_count = 0
        self.__frame_rate = 0
        self.dims = (0, 0, 0)
        self.__open(path)
        self.__path = path
        self.__output_path = ''
        self.__detector = PoseDetector(Input.VIDEO, EvalType.FULL)
        self.__frame_buffer = FrameBuffer(self.__frame_count, self.dims,
                                          maxsize=2048, lock=True)
        self.__video_completed = threading.Event()

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
        input_file_name = os.path.splitext(os.path.basename(self.__path))[0]
        self.__output_path = os.path.dirname(__file__)
        self.__output_path = os.path.join(self.__output_path, f'../output/{input_file_name}_analyzed.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(self.__output_path, fourcc, 30, (self.dims[1], self.dims[0]))
        playback = False
        lost_frames = 0
        counter = 0
        while True:
            if self.__frame_buffer.current_frames() > 0:
                start = time.time()
                frame = Frame(self.__frame_buffer.pop())
                playback = True
                if frame is None:
                    print("Empty frame received")
                    break
                counter += 1
                mp_image = frame.to_mediapipe_image()
                res = self.__detector.get_body_key_points(mp_image, counter)
                if res.pose_landmarks:
                    frame.annotate(res.pose_landmarks, as_overlay=True)
                    end = time.time()
                    fps = 1 / (end - start)
                    cv2.putText(frame.data(), f'FPS: {fps:.2f}', (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    out.write(frame.data())
                # cv2.imshow("TEST", frame)
                if cv2.waitKey(0) & 0xFF == ord('q'):
                    break
            elif playback and self.__video_completed.is_set():
                break
        out.release()
        lost_frames = (1 - (counter / self.__frame_count)) * 100
        print(f"""video {self.__path} analyzed \n
              lost frames: {self.__frame_count - counter} 
              ({lost_frames:.2f}%)""")
        cv2.destroyAllWindows()

    def analyze(self):
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
