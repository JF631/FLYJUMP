'''
Module that provides a thread safe ring buffer that is used to store video 
frames or live stream frames.

Writing and reading operations can be synchronized using a conditional lock.

Author: Jakob Faust (software_jaf@mx442.de)
Date: 2023-10-20
'''

import threading
import warnings
import psutil
import numpy as np

from utils.warnings import WarningDialog, PotentialRaceConditionWarning

class FrameBuffer():
    '''
    Abstraction of a ring buffer with thread safe add() and pop() methods.
    It supports simultaneous reading and writing operations. We use it in 
    combination with two threads:
        1) Read video frame by frame from file or drone stream, visualize the 
        frame and safe it to this buffer.
        
        2) Read from the same buffer and perform pose detection.
        
        
    As the write operation will likely be quicker than the pose detection 
    operation, race conditions may occur.
    That's why a conditional lock is used as synchronization primitive.
    
    Parameters
    ----------
    frame_count : int
        number of frames present in video or number of frames needs to be 
        buffered when using in combination with live stream.
    frame_props : tuple
        frame size in (width, height, channels)
    lock : bool
        enable (default) or disable the synchronization.
        if disabled, you have to handle lost frames or ensure a large enough 
        buffer, so that the writing thread cannot catch up and overwrite data
        that the processing thread has not processed yet.
    
    Usage hint
    -----------
    The larger this buffer is, the quicker the writing thread will be able to 
    work.
    The ringbuffer itself takes care about not allocating too much memory. 
    
    CAUTION
    --------
    Because of synchronization the visualizing thread might be slowed down
    significantly.
    Especially for small buffer sizes, the first thread has to wait for the 
    second thread whenever it catches up. 
    '''
    def __init__(self, frame_count:int,
                 frame_dims:tuple, maxsize:int=2048, lock:bool=True) -> None:
        self.max_size = maxsize # must be power of 2
        self.frame_dims = frame_dims
        self.frame_count = frame_count
        self.size = self.__ensure_memory(self.max_size)
        self.warning_dialog = WarningDialog()
        warnings.showwarning = self.warning_dialog.show_warning
        if self.size < self.frame_count and not lock:
            warnings.warn(f"""Buffer lock is disabled and buffer size
                          {self.size} is smaller than the frame count
                          {self.frame_count} - This might lead to race 
                          conditions!""", PotentialRaceConditionWarning)
        # self.frame_buffer = [None] * self.size
        self.frame_buffer = np.empty((self.size, *self.frame_dims), dtype='u1')
        self.current_end = 0
        self.current_start = 0
        self.enable_lock = lock
        if self.enable_lock:
            self.lock = threading.Lock()
            self.condition = threading.Condition(self.lock)

    def __next_convenient_size(self, size:int):
        '''
        Finds next power of 2 that is larger or equal to the input size.

        Parameters
        ----------
        size : int
            video frame count or desired buffer size.
        '''
        if size & (size - 1) == 0: # size is already power of 2
            return size
        rtrn = 1
        while rtrn < size:
            rtrn <<= 2
        return rtrn

    def __ensure_memory(self, max_size):
        '''
        Ensures that the buffer does not exceed the available memory.
        if desired buffer size is larger than available memory, the buffer size
        is iteratively reduced by a factor of 2 until it fits into memory.

        Parameters
        ----------
        max_size : int
            guaranteed that the buffer won't exceed this size (in elements).    
        '''
        rtrn = (
            min(self.__next_convenient_size(self.frame_count), max_size))
        required_memory = (rtrn * self.frame_dims[0] *
                           self.frame_dims[1] * self.frame_dims[2])
        available_memory = psutil.virtual_memory().available
        # print(f"Frame buffer of size: {rtrn} initialized, \n \
        #         required memory: {required_memory / 1024 /  1024} MB, \n \
        #         {required_memory / available_memory * 100} % of total memory")
        if required_memory > available_memory:
            print("no more memory - trying to downsize buffer")
            return self.__ensure_memory(max_size >> 2)
        return rtrn


    def add(self, frame: np.ndarray):
        '''
        Adds a frame to the ring buffer based on FIFO principle.

        If synchronization is enabled, the opperation is delayed until the
        processing thread has processed the frame that is about to be 
        overwritten.

        Parameters
        ----------
        frame : numpy.ndarray
            frame to be added to the buffer (width, height, channels).
        '''
        if self.enable_lock:
            with self.lock:
                while self.current_frames() == self.size - 1:
                    self.condition.wait()
                self.frame_buffer[self.current_end] = frame
                self.current_end = (self.current_end + 1) % self.size
                self.condition.notify()
        else:
            self.frame_buffer[self.current_end] = frame
            self.current_end = (self.current_end + 1) % self.size

    def pop(self):
        '''
        Returns the oldest frame that has not been processed yet from the 
        buffer.

        pop() is chosen as name to indicate that the frame is invalid after
        this operation as it will eventually be overwritten by the writing 
        thread - it is NOT actually deleted from memory.

        If synchronization is enabled, the opperation is delayed until a new 
        frame is available. In reality, this shouldn't happen too often as the
        writing thread will (always) be faster than the processing thread.

        '''
        if self.enable_lock:
            with self.lock:
                while self.current_frames() == 0:
                    self.condition.wait()
                rtrn = self.frame_buffer[self.current_start]
                self.current_start = (self.current_start + 1) % self.size
                self.condition.notify()
                return rtrn
        else:
            rtrn = self.frame_buffer[self.current_start]
            self.current_start = (self.current_start + 1) % self.size
            return rtrn

    def get_frame(self, index:int):
        '''
        Returns frame for a given index.

        Parameters
        ----------
        index : int
            index of frame that should be returned.
            Must be in range [0, self.size)

        Returns
        -------
        frame : np.ndarray
            requested frame if it is present, None otherwise.

        '''
        if(index < 0 or index >= self.size):
            raise IndexError(f"""index {index} out of bounds - must be between
                                0 and {self.size}""")
        return self.frame_buffer[index]

    def current_frames(self):
        '''
        Keeps track of the total number of VALID frames in buffer.
        This is especially  important when synchronization in enabled.

        Returns
        -------
        num_valid : int
            number of valid frames currently present in the buffer. 
        '''
        if self.current_start <= self.current_end:
            return self.current_end - self.current_start
        return (self.size - self.current_start) + self.current_end

    def get_latest(self):
        '''
        Returns last valid frame in buffer.

        Returns
        -------
        frame : np.ndarray
            last valid frame in buffer, shape (width, height, channels)
        '''
        return self.get_frame(self.current_end)

    def get_size(self):
        '''
        Returns number of elements (frames) the buffer is able store
        
        Returns
        -------
        size : int
            buffer size in elements
        '''
        return self.size

    def __iter__(self):
        self.iter_index = 0
        return self

    def __next__(self):
        if self.iter_index < self.size:
            self.iter_index += 1
            return self.frame_buffer[self.iter_index]
        raise StopIteration
