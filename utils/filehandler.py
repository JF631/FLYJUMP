'''
Module for handling file export / import operations.

Author: Jakob Faust (software_jaf@mx442.de)
Date: 2023-11-07
'''

import os
from datetime import datetime

import h5py
import numpy as np

from ljanalyzer.frame import Frame

class ParameterFile():
    '''
    A parameter file holds analysis results for each frame.
    It is saved on disk as hdf5 file.
    '''
    def __init__(self, file_path:str, signals = None) -> None:
        self.__frame_data = {}
        self.__frame_count = 0
        self.__batchsize = 128
        self.__file_path = file_path
        self._hip_height = []
        self._right_foot_pos = []
        self._left_foot_pos = []
        if signals:
            signals.error.connect(self.close)

    def __add_to_dict(self, frame: Frame):
        '''
        Adds frame to temporary buffer.
        
        Parameters
        ----------
        frame : Frame
            frame object whose parameters should be saved to disk
        '''
        r_knee_angle, l_knee_angle = frame.knee_angles()
        l_foot_x, l_foot_y, r_foot_x, r_foot_y = 1 - frame.foot_pos().ravel(order='F')
        hip_height = 1- frame.centroid_height()
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

    def get_path(self):
        '''
        get file path of current parameter file.

        Returns
        --------
        path : str
            absolute file path of current parameter file.
        '''
        return self.__file_path

    def __write_to_file(self):
        '''
        Writes all frame parameters from buffer to disk
        '''
        file_name = self.get_path()
        print(f"writing to {file_name}")
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
    
    def get_video_path(self):
        '''
        returns video path to current analysis file.
        Assumes that the video is in the same directory as the analysis file.
        
        Returns
        -------
        path : str
            output analysis file path
            (e.g. 'C:/FLYJUMP/analysis/2023-01-01/vid0.mp4').
        '''
        return os.path.splitext(self.__file_path)[0] + '.mp4'

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
    
    def add_metadata(self, data:tuple):
        '''
        saves metadata in the root of the hdf5 file.

        Parameters
        ----------
        data : tuple
            (key, metadata) - metadata that should be stored under 'key'
        '''
        file_name = self.get_path()
        key, metadata = data
        with h5py.File(file_name, 'a') as param_file:
            param_file.attrs[key] = metadata

    def get_takeoff_frame(self)->int:
        '''
        get takeoff frame number from parameter file.
        
        Returns
        -------
        takeoff_frame : int
            number in which the takeoff has been detected
        '''
        file_name = self.get_path()
        rtrn = -1
        with h5py.File(file_name, 'a') as param_file:
            rtrn = param_file.attrs.get('takeoff', -1)
        return rtrn

    def close(self):
        '''
        Saves all remaining frames to file.
        
        Is called when a video object emits finished signal. 
        '''
        print("saving last data..")
        self.__write_to_file()
    
    def load(self):
        '''
        loads parameter file from disk.
        fills following lists:
        - right_foot_height
        - left_foot_height
        - hip height
        
        The data in the above lists is ordered frame-wise.
        The lists are accessible via own functions (see below).
        
        '''
        right_foot_y = []
        left_foot_y = []
        hip_height = []
        with h5py.File(self.get_path(), 'r') as param_file:
            '''
            following line: param_file.keys() returns sth. like 'frame_10'
            sorting key will then be int(frame_10[6:]) resulting in 10.
                                               ^
                                      this is the 6th index
            '''
            sorted_key = sorted(param_file.keys(), key=lambda x : int(x[6:]))
            for frame in sorted_key:
                group = param_file[frame]
                right_y = group["right_foot_y"][()]
                right_foot_y.append(right_y)
                left_y = group["left_foot_y"][()]
                left_foot_y.append(left_y)
                hip_y = group["hip_height"][()]
                hip_height.append(hip_y)
            self._hip_height = np.array(hip_height)
            self._left_foot_pos = np.array(left_foot_y)
            self._right_foot_pos = np.array(right_foot_y)
    
    def get_hip_height(self):
        '''
        Relative hip height over time.

        Returns
        -------
        hip_height : np.ndarray
            hip height over time as flat numpy array
            shape (num_frames,)
        '''
        return self._hip_height
    
    def get_right_foot_height(self):
        '''
        Relative hip height over time.

        Returns
        -------
        hip_height : np.ndarray
            hip height over time as flat numpy array
            shape (num_frames,)
        '''
        return self._right_foot_pos
    
    def get_left_foot_height(self):
        '''
        Relative hip height over time.

        Returns
        -------
        hip_height : np.ndarray
            hip height over time as flat numpy array
            shape (num_frames,)
        '''
        return self._left_foot_pos

class FileHandler():
    '''
    The Filehandler takes care to create the correct folder structure and
    provides all necessary paths to interact with parameter files.

    Outputs can be found under:
    software_path/analysis/YYYY-MM-dd
    Where date is the ANALYSIS DATE
    '''
    __OUTPUT_FOLDER = "analysis"
    def __init__(self) -> None:
        pass

    @staticmethod
    def create_general_structure() -> bool:
        '''
        creates "analysis" folder, that later holds all analysis outputs.
        '''
        analysis_output = os.path.join(os.getcwd(),
                                       FileHandler.__OUTPUT_FOLDER)
        if not os.path.exists(analysis_output):
            os.makedirs(analysis_output)

    @staticmethod
    def get_output_path() -> str:
        '''
        Get absolut output path.

        Returns
        -------
        absolute path to the software's output folder in which
        anlysis results are stored.
        '''
        return os.path.join(os.getcwd(), FileHandler.__OUTPUT_FOLDER)

    @staticmethod
    def create_current_folder() -> str:
        '''
        creates folder inside analysis output folder with current
        date.
        Analysis outputs are actually stored like:
        software_path/analysis/YYYY-MM-dd
        
        Returns
        -------
        Either newly created folder path or existing path (if already created
        before)
        '''
        current_date = datetime.now().strftime("%Y-%m-%d")
        path = os.path.join(os.getcwd(), FileHandler.__OUTPUT_FOLDER
                            + f'/{current_date}')
        if not os.path.exists(path):
            os.makedirs(path)
        return path
