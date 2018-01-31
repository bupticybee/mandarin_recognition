import signal
import numpy as np
import traceback
from multiprocessing import Queue, Process
import time

class async_flow(object):
    def producer(self):
        if self._shuffle:
            rand_indexes = np.random.choice(len(self._dataflow), len(self._dataflow), replace=False)
        else:
            rand_indexes = np.arange(len(self._dataflow))
        iterator_index = 0
        
        while(True):
            try:
                if self._q.qsize() < self._max_buffer_batch:
                    indexes = rand_indexes[iterator_index: iterator_index + self._batch_size]
                    time.sleep(0.001)
                    if not( self._small_batch == False and len(indexes) < self._batch_size):
                        self._q.put(self._dataflow[indexes])
                    iterator_index += self._batch_size
                    if iterator_index >= len(self._dataflow):
                        if self._shuffle:
                            rand_indexes = np.random.choice(len(self._dataflow), len(self._dataflow), replace=False)
                        iterator_index = 0
                else:
                        time.sleep(0.001)
            except Exception as e: 
                time.sleep(1)
                traceback.print_exc()
        
    def __init__(self, dataflow, shuffle = True, max_buffer_batch = 4, batch_size = 32,small_batch=False):
        self._small_batch = small_batch
        self._dataflow = dataflow
        self._shuffle = shuffle
        self._max_buffer_batch = max_buffer_batch
        self._batch_size = batch_size
        self._q = Queue()
        self._p1 = Process(target=self.producer, args=())
        self._p1.start()
        
        
    def next_batch(self,bs):
        while(self._q.qsize() <= 2):
            time.sleep(0.001)
        ret_values = self._q.get()
        return ret_values
    
    def buffer_size(self):
        return self._q.qsize()
    
    def stop(self):
        self._p1.terminate()
