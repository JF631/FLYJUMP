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

    def close(self):
        '''
        Saves all remaining frames to file.
        
        Is called when a video object emits finished signal. 
        '''
        print("saving last data..")
        self.__write_to_file()
        # self.load(file_name)
    
    def load(self):
        right_foot_y = []
        left_foot_y = []
        hip_height = []
        with h5py.File(self.get_path(), 'r') as param_file:
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

    # @staticmethod
    # def load(path: str):
    #     right_foot_y = []
    #     left_foot_y = []
    #     hip_height = []
    #     with h5py.File(path, 'r') as param_file:
    #         sorted_key = sorted(param_file.keys(), key=lambda x : int(x[6:]))
    #         for frame in sorted_key:
    #             group = param_file[frame]
    #             right_y = group["right_foot_y"][()]
    #             right_foot_y.append(right_y)
    #             left_y = group["left_foot_y"][()]
    #             left_foot_y.append(left_y)
    #             hip_y = group["hip_height"][()]
    #             hip_height.append(hip_y)
    #     total_error = 100
    #     index = 0
    #     possible_indices = []
    #     runup_coeffs = []
    #     jump_coeffs = []
    #     for i in range(2, len(hip_height) - 2):
    #         x_runup = np.arange(len(hip_height[:i]))
    #         x_jump = np.arange(len(hip_height[i:]))
    #         hip_fit_runup, residuals_runup, _, _, _ = np.polyfit(
    #             x_runup,hip_height[:i], 1, full=True)
    #         hip_fit_jump, residuals_jump, _, _, _ = np.polyfit(
    #             x_jump, hip_height[i:], 2, full=True)
    #         fitting_error = residuals_runup + residuals_jump
    #         if fitting_error:
    #             possible_indices.append(fitting_error[0])
    #         if fitting_error < total_error:
    #             total_error = fitting_error
    #             index = i
    #             runup_coeffs = hip_fit_runup
    #             jump_coeffs = hip_fit_jump
    #     possible_indices = np.array(possible_indices)
    #     possible_indices = np.where(np.logical_and(
    #         (possible_indices < (total_error + total_error*0.01)), 
    #         (possible_indices > (total_error - total_error*0.01))))[0]
    #     '''
    #     TODO verify the following if statement makes sense and is reasonable.
    #     It is meant to solve the problem that a takeoff is detected during the
    #     runup.
    #     '''
    #     if(index < possible_indices[-1]):
    #         index = possible_indices[-1]
    #     print(f"takeoff detected at {index}")
    #     hip_runup = np.poly1d(runup_coeffs)
    #     hip_jump = np.poly1d(jump_coeffs)
    #     x_runup = np.arange(0, index)
    #     x_jump = np.arange(index, len(hip_height))
    #     plt.xlabel("t [frames]")
    #     plt.ylabel("height [norm. pix]")
    #     plt.plot(hip_height, label='hip')
    #     plt.plot(x_runup, hip_runup(x_runup), label="runup")
    #     plt.plot(x_jump, hip_jump(np.arange(len(x_jump))), label="jump")
    #     plt.plot(left_foot_y, label='left foot')
    #     plt.plot(right_foot_y, label='right foot')
    #     plt.legend()
    #     file_name ='test.png'
    #     plt.savefig(file_name)

class FileHandler():
    '''
    The Filehandler takes care to create the correct folder structure and
    provides all neccessarry paths to interact with parameter files.

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
        '''
        current_date = datetime.now().strftime("%Y-%m-%d")
        path = os.path.join(os.getcwd(), FileHandler.__OUTPUT_FOLDER
                            + f'/{current_date}')
        if not os.path.exists(path):
            os.makedirs(path)
        return path
