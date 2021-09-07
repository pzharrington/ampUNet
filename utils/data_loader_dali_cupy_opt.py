import torch
import numpy as np
import cupy as cp
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch import Tensor
import h5py

#concurrent futures
import concurrent.futures as cf

#dali stuff
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy


def get_data_loader_distributed(params, world_rank, device_id = 0):
    train_loader = DaliDataLoader(params, params.train_path, params.Nsamples, num_workers=params.num_data_workers, device_id=device_id)
    validation_loader = DaliDataLoader(params, params.validation_path, params.Nsamples_val, num_workers=params.num_data_workers, device_id=device_id)
    return train_loader


class DaliInputIterator(object):
    def pin(self, array):
        mem = cp.cuda.alloc_pinned_memory(array.nbytes)
        ret = np.frombuffer(mem, array.dtype, array.size).reshape(array.shape)
        ret[...] = array
        return ret
                            
    def __init__(self, params, data_file, num_samples, device_id):
        # set device
        self.device_id = device_id
        cp.cuda.Device(self.device_id).use()
        
        # memory pool
        self.pinned_memory_pool = cp.cuda.PinnedMemoryPool()
        cp.cuda.set_pinned_memory_allocator(self.pinned_memory_pool.malloc)
        
        # stream
        self.stream_htod = cp.cuda.Stream(non_blocking=True)
        
        # stage in data
        self.hfile = h5py.File(data_file, 'r')
        self.fields_inp = self.hfile['Nbody']
        self.fields_tar = self.hfile['Hydro']
        
        # size of dataset
        self.n = num_samples
        
        # other parameters
        self.batch_size = params.batch_size
        self.data_file = data_file
        
        # other params
        self.inp_buff = None
        self.tar_buff = None

        # length
        self.length =  params.box_size
        self.size = params.data_size

        # shapes
        self.inp_shape_full = (4, self.length, self.length, self.length)
        self.tar_shape_full = (5, self.length, self.length, self.length) 
        self.inp_shape = (4, self.size, self.size, self.size)
        self.tar_shape = (5, self.size, self.size, self.size)
        
        # CPU
        self.inp_buff_cpu = self.pin(np.zeros((self.batch_size,) + self.inp_shape, dtype=np.float32))
        self.tar_buff_cpu = self.pin(np.zeros((self.batch_size,) + self.tar_shape, dtype=np.float32))
        
        # GPU
        self.inp_buff_gpu = [cp.zeros((self.batch_size,) + self.inp_shape, dtype=np.float32),
                             cp.zeros((self.batch_size,) + self.inp_shape, dtype=np.float32)]
        self.tar_buff_gpu = [cp.zeros((self.batch_size,) + self.tar_shape, dtype=np.float32),
                             cp.zeros((self.batch_size,) + self.tar_shape, dtype=np.float32)]

        # rng
        self.rng = np.random.RandomState(seed=12345)

        # index list
        self.indices = list(range(self.n))
        self.rng.shuffle(self.indices)
        self.i = 0
        
        # thread pool
        self.executor = cf.ThreadPoolExecutor(max_workers = 2)
        
        # submit data fetch
        self.curr_buff = 0
        buff_ind = self.curr_buff
        self.future = self.executor.submit(self.get_batch, buff_ind)

        
    def __iter__(self):
        self.i = 0
        self.rng.shuffle(self.indices)
        return self
        

    def get_batch(self, buff_id):
        # set device
        cp.cuda.Device(self.device_id).use()
        
        # mark region
        torch.cuda.nvtx.range_push("DaliInputIterator:get_batch")

        # generate coordinates
        coords = self.rng.randint(low=0, high=self.length-self.size, size=(self.batch_size, 3))
        
        # read
        for batch_id in range(self.batch_size):

            # extract coords
            x, y, z = coords[batch_id, 0], coords[batch_id, 1], coords[batch_id, 2]
        
            # inp
            self.fields_tilde.read_direct(self.inp_buff_cpu,
                                          np.s_[0:4, x:x+self.size, y:y+self.size, z:z+self.size],
                                          np.s_[0:4, 0:self.size, 0:self.size, 0:self.size])
                
        # upload
        self.inp_buff_gpu[buff_id].set(self.inp_buff_cpu, self.stream_htod)
                

        for batch_id in range(self.batch_size):

            # extract coords
            x, y, z = coords[batch_id, 0], coords[batch_id, 1], coords[batch_id, 2]

            # target
            self.fields_hr.read_direct(self.tar_buff_cpu,
                                       np.s_[0:5, x:x+self.size, y:y+self.size, z:z+self.size],
                                       np.s_[0:5, 0:self.size, 0:self.size, 0:self.size])
                    
        # upload
        self.tar_buff_gpu[buff_id].set(self.tar_buff_cpu, self.stream_htod)
        
        # synchronize
        self.stream_htod.synchronize()
        
        # create handles
        inp = self.inp_buff_gpu[buff_id]
        tar = self.tar_buff_gpu[buff_id]
        
        # finish region
        torch.cuda.nvtx.range_pop()
        
        return inp, tar

        
    def __next__(self):
        torch.cuda.nvtx.range_push("DaliInputIterator:next")
        # wait for batch load to complete
        if self.future is None:
            raise StopIteration
            
        inp, tar = self.future.result()
        
        # submit new work before proceeding
        # increase batch counter
        self.i += self.batch_size
        
        # adjust current buffer
        self.curr_buff = (self.curr_buff + 1) % 2
        buff_ind = self.curr_buff
        
        # submit work if epoch not done
        if self.i + self.batch_size < self.n:
            self.future = self.executor.submit(self.get_batch, buff_ind)
        else:
            self.future = None
        torch.cuda.nvtx.range_pop()
        
        return inp, tar

    next = __next__

    
class DaliDataLoader(object):
    """Random crops"""
    def get_pipeline(self, params, data_file, num_samples, num_workers, device_id):

        # construct master object
        pipeline = Pipeline(batch_size = params.batch_size,
                            num_threads = num_workers,
                            device_id = device_id)

        # construct ES
        dii = DaliInputIterator(params, data_file, num_samples, device_id)

        with pipeline:
            data, label = fn.external_source(source = dii,
                                             num_outputs = 2,
                                             cycle = "raise",
                                             no_copy = True,
                                             parallel = False)

            # copy to gpu: not necessary
            #data = data.gpu()
            #label = label.gpu()

            # return
            pipeline.set_outputs(data, label)

        return pipeline
    
    def __init__(self, params, data_file, num_samples, num_workers=1, device_id=0):
        # construct pipeline
        self.pipe = self.get_pipeline(params, data_file, num_samples, num_workers, device_id)
        self.pipe.build()
        
        self.iterator = DALIGenericIterator([self.pipe], ['inp', 'tar'],
                                            size = -1,
                                            last_batch_policy = LastBatchPolicy.PARTIAL,
                                            auto_reset = True,
                                            prepare_first_batch = True)
        
    def __len__(self):
        return self.length
        
    def __iter__(self):
        for token in self.iterator:
            inp = token[0]['inp']
            tar = token[0]['tar']
            yield inp, tar
