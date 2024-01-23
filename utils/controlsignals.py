'''
Module that holds relevant objects for the overalll software behavior.

Author: Jakob Faust (software_jaf@mx442.de)
Date: 2023-11-07
'''
from threading import Lock
from PyQt5.QtCore import QObject, pyqtSignal

class SharedBool:
    '''
    Tries to immitate C++'s std::atomic<bool>.

    It is used for interrupting running threads.
    A reference of the SharedBool is simply passed to the QRunnable thread.
    Whenever the set() method is called, all running threads will try to 
    terminate. 
    '''
    def __init__(self) -> None:
        self.__lock = Lock()
        self.__value = False

    def set(self):
        '''
        Sets value to True. 
        '''
        with self.__lock:
            self.__value = True

    def reset(self):
        with self.__lock:
            self.__value = False

    def get(self)->bool:
        '''
        Returns
        -------
        interrupt : bool
            if True interrupt threads, else threads run normally
        '''
        with self.__lock:
            return self.__value

class ControlSignals(QObject):
    '''
    Defines PyQt Signals that are used to control the overall software flow.

    Signals
    --------
    terminate : pyqtSignal
        gracefully terminate all running processes - release buffers and close
        programm
    jump_to_frame : pyqtSignal
        emmitted whenever the video playback should jump to a certain frame.
    '''
    terminate = pyqtSignal()
    jump_to_frame = pyqtSignal(int)
