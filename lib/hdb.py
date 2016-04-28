#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""Greeter.

Usage:
  hdb.py generate <maxsize> [--gzip] [--lzf] [<chunk>]
  hdb.py test <maxsize> 
  hdb.py -h | --help

Options:
  -h --help     Show this screen.
""" 
import h5py
import os.path 

class Database:
    count=0
    maxsize=1000
    shape=(90,90,1)
    name="mydataset"
    initialized=False
self=Database

def __init__(maxsize=2000000,shape=(90,90,1),reset=False,file_path='h.hdf5',
        compression=None,
        chunk_size=None):
    self.maxsize=maxsize
    self.shape=shape
    if os.path.exists(file_path) and not reset:
        f=h5py.File(file_path)
        self.dset = f.get(self.name)
    else:
        f=h5py.File(file_path,'w')
        if not None ==chunk_size:
            chunk_size=(chunk_size,)+shape
        
        self.dset = f.create_dataset(self.name,
                (self.maxsize,)+shape,
                chunks=chunk_size,
                compression=compression,
                dtype='float32')
def save(arr):
    if not self.initialized:
        self.initialized=True
        __init__(shape=arr.shape,reset=True)
    i0=self.count%self.maxsize
    self.count+=1
    self.dset[i0]=arr
    return i0
def load(i0):
    arr=self.dset[i0]
    return arr
    

if '__main__' == __name__:
    import numpy as np
    from docopt import docopt
    arguments = docopt(__doc__)
    maxsize=int(arguments['<maxsize>'])
    if arguments['generate']:
        #生成大数据文件
        if arguments['--gzip']:
            compression='gzip'
        elif arguments['--lzf']:
            compression='lzf'
        else:
            compression=None
        chunk_size=arguments['<chunk>']
        if not None==chunk_size:
            chunk_size=int(chunk_size)

        import time
        import random
        t0=time.time()
    #    __init__(maxsize,shape=(90,90,2),reset=True,compression=compression,chunk_size=chunk_size) 
       # a=np.load('/dev/shm/depth1ds.npy')
        a=np.zeros((90,90))
        for i in range(10):
            save(a+i)
        t1=time.time()
        print('time:%s'%(t1-t0))
    elif arguments['test']:
        #测试随机读取2000图片的性能
        __init__(maxsize)
        import time
        import random
        t0=time.time()
        l=[]
        for _ in range(2000):
            i=int(random.random()*maxsize)
            j=load(i)
            l.append(i-np.mean(j)) 
        print(max(l),min(l))
        t1=time.time()
        print('time:%s'%(t1-t0))
        '''
        hdb.py generate 1000
        hdb.py test 1000
        time:0.221511125565
        hdb.py generate 10000
        hdb.py test 10000
        time:0.755643844604
        hdb.py test 10000
        time:0.906985044479
        hdb.py generate 100000
        hdb.py test 100000
        time:18.4068500996
        '''
