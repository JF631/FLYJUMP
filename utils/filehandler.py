'''
Module for handling file export / import operations.

Author: Jakob Faust (software_jaf@mx442.de)
Date: 2023-11-07
'''

import os
from datetime import datetime

import h5py

from ljanalyzer.frame import Frame
import matplotlib.pyplot as plt
import numpy as np

class ParameterFile():
    def __init__(self, signals, file_name:str) -> None:
        self.__frame_data = {}
        self.__frame_count = 0
        self.__batchsize = 128
        self.__filename = FileHandler.create_current_folder()
        self.__filename = os.path.join(self.__filename, 
                                       f'{file_name}')
        signals.finished.connect(self.__save_last)

    def __add_to_dict(self, frame: Frame):
        '''
        Adds frame to temporary buffer.
        
        Parameters
        ----------
        frame : Frame
            frame object whose parameters should be saved to disk
        '''
        r_knee_angle, l_knee_angle = frame.knee_angles()
        l_foot_x, l_foot_y, r_foot_x, r_foot_y = frame.foot_pos().ravel(order='F')
        hip_height = frame.centroid_height()
        frame_key = f"frame_{self.__frame_count}"
        self.__frame_data[frame_key] = {
            "right_knee_angle" : r_knee_angle,
            "left_knee_angle": l_knee_angle,
            "right_foot_x" : r_foot_x,
            "right_foot_y" : r_foot_y,
            "left_foot_x" : l_foot_x,
            "left_foot_y" : l_foot_y,
            "hip_height" : hip_height
        }
        self.__frame_count += 1

    def __write_to_file(self):
        '''
        Writes all frame parameters from buffer to disk
        '''
        file_name = self.__filename + '.hdf5'
        with h5py.File(file_name, 'a') as param_file:
            for frame_key, data in self.__frame_data.items():
                frame_group = param_file.create_group(frame_key)
                frame_group.create_dataset("right_knee_angle", 
                                            data=data['right_knee_angle'])
                frame_group.create_dataset("left_knee_angle", 
                                            data=data['left_knee_angle'])
                frame_group.create_dataset("right_foot_x", 
                                            data=data['right_foot_x'])
                frame_group.create_dataset("right_foot_y", 
                                            data=data['right_foot_y'])
                frame_group.create_dataset("left_foot_x", 
                                            data=data['left_foot_x'])
                frame_group.create_dataset("left_foot_y", 
                                            data=data['left_foot_y'])
                frame_group.create_dataset("hip_height", 
                                            data=data['hip_height'])
        self.__frame_data.clear()

    def save(self, frame: Frame):
        '''
        Adds frame to temporary buffer and writes it in batches of size 128 to 
        disk.
        
        Parameters
        ----------
        frame : Frame
            frame object whose parameters should be saved to disk
        '''
        self.__add_to_dict(frame)
        if self.__frame_count % self.__batchsize == 0:
            self.__write_to_file()

    def __save_last(self):
        '''
        Saves all remaining frames to file.
        
        Is called when a video object emits finished signal. 
        '''
        print("saving last data..")
        self.__write_to_file()
        file_name = self.__filename + '.hdf5'
        # self.load(file_name)
  
    def load(self, path: str):
        right_foot_y = []
        left_foot_y = []
        hip_height = []
        with h5py.File(path, 'r') as param_file:
            sorted_key = sorted(param_file.keys(), key=lambda x : int(x[6:]))
            for frame in sorted_key:
                # print(frame)
                group = param_file[frame]
                right_y = group["right_foot_y"][()]
                right_foot_y.append((1.0 - right_y))
                left_y = group["left_foot_y"][()]
                left_foot_y.append((1.0 - left_y))
                hip_y = group["hip_height"][()]
                hip_height.append((1.0 - hip_y))
                # print(f"r: {right_y}, l: {left_y}")

        window_size = 3
        smoothed_hip = np.convolve(hip_height, np.ones(window_size) / window_size, mode='same')
        smoothed_left = np.convolve(left_foot_y, np.ones(window_size) / window_size, mode='same')
        smoothed_right = np.convolve(right_foot_y, np.ones(window_size) / window_size, mode='same')
        plt.xlabel("t [frames]")
        plt.ylabel("height [norm. pix]")
        plt.plot(hip_height, label='hip')
        plt.plot(left_foot_y, label='left foot')
        plt.plot(right_foot_y, label='right foot')
        plt.legend()
        file_name = self.__filename + '.png'
        plt.savefig(file_name)
        # plt.show()

class FileHandler():
    __OUTPUT_FOLDER = "analysis"
    def __init__(self) -> None:
        pass

    def create_general_structure() -> bool:
        '''
        creates "analysis" folder, that later holds all analysis outputs.
        '''
        analysis_output = os.path.join(os.getcwd(), 
                                       FileHandler.__OUTPUT_FOLDER)
        if not os.path.exists(analysis_output):
            os.makedirs(analysis_output)

    def get_output_path() -> str:
        return os.path.join(os.getcwd(), FileHandler.__OUTPUT_FOLDER)
    
    def create_current_folder() -> str:
        current_date = datetime.now().strftime("%Y-%m-%d")
        path = os.path.join(os.getcwd(), FileHandler.__OUTPUT_FOLDER
                            + f'/{current_date}')
        if not os.path.exists(path):
            os.makedirs(path)
        return path
