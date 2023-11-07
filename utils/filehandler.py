'''
Module for handling file export / import operations.

Author: Jakob Faust (software_jaf@mx442.de)
Date: 2023-11-07
'''

import os

import h5py

from ljanalyzer.frame import Frame

class ParameterFile():
    def __init__(self, signals, file_name:str) -> None:
        self.__frame_data = {}
        self.__frame_count = 0
        self.__batchsize = 128
        self.__filename = os.path.dirname(__file__)
        self.__filename = os.path.join(self.__filename, 
                                       f'../output/{file_name}.hdf5')
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
        frame_key = f"frame{self.__frame_count}"
        self.__frame_data[frame_key] = {
            "right_knee_angle" : r_knee_angle,
            "left_knee_angle": l_knee_angle 
        }
        self.__frame_count += 1

    def __write_to_file(self):
        '''
        Writes all frame parameters from buffer to disk
        '''
        with h5py.File(self.__filename, 'a') as param_file:
            for frame_key, data in self.__frame_data.items():
                frame_group = param_file.create_group(frame_key)
                frame_group.create_dataset("right_knee_angle", 
                                            data=data['right_knee_angle'])
                frame_group.create_dataset("left_knee_angle", 
                                            data=data['left_knee_angle'])
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
    
    def load(self, path: str):
        pass