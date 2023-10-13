import threading

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
    size : int
        size of ringbuffer in elements.
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
    def __init__(self, size:int, lock:bool=True) -> None:
        self.size = size
        self.frame_buffer = [None] * size
        self.current_end = 0 
        self.current_start = 0 
        self.enable_lock = lock
        if self.enable_lock:
            self.lock = threading.Lock()
            self.condition = threading.Condition(self.lock)
        
    def add(self, frame):
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
        
    def get_frame(self, index):
        return self.frame_buffer[index]
    
    def pop(self):
        # print(f"end {self.current_end}, start {self.current_start}, frames left: {self.current_frames()}")
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
    
    def current_frames(self):
        if self.current_start <= self.current_end:
            return self.current_end - self.current_start
        else:
            return (self.size - self.current_start) + self.current_end
    
    def get_latest(self):
        return self.get_frame(self.current_end)
    
    def get_size(self):
        return self.size
    
    def __iter__(self):
        self.iter_index = 0
        return self
    
    def buffer_filled(self):
        self.current_end -= 1
    
    def __next__(self):
        if self.iter_index < self.size:
            self.iter_index += 1
            return self.frame_buffer[self.iter_index]
        raise StopIteration
