import cv2
import numpy as  np
import os.path as osp
import mmap

from .shm import *
from decimal import Decimal, ROUND_HALF_UP

import logging


def imread(image_path, mode=1, dtype=np.uint8):

#    with shm_open(image_path) as fd:
    fd = shm_open(image_path)
    try:
        msize = 0
        with mmap.mmap(fd, 4) as mm:
            msize = mm.read_byte()
            msize = msize | (mm.read_byte() << 8)
            msize = msize | (mm.read_byte() << 16)
            msize = msize | (mm.read_byte() << 24)
        
        with mmap.mmap(fd, msize) as mm:
            mm.seek(4)
            img = cv2.imdecode(np.frombuffer(mm.read(), dtype=dtype), mode)
            #return cv2.imdecode(np.frombuffer(mm.read(), dtype=dtype), mode)
    except:
        logging.error('Error openMemory')
        img = cv2.imread(image_path)
        #return cv2.imread(image_path)
    
    shm_close(fd)
    return img

    

'''
#【Xavierポーティング】
    try:
        fn = image_path
        msize = 0
        with mmap.mmap(-1, 4, fn) as mm:
            msize = mm.read_byte()
            msize = msize | (mm.read_byte() << 8)
            msize = msize | (mm.read_byte() << 16)
            msize = msize | (mm.read_byte() << 24)
        with mmap.mmap(-1, msize, fn) as mm:
            mm.seek(4)
            return cv2.imdecode(np.frombuffer(mm.read(), dtype=dtype), mode)
    except:
        logging.error('Error openMemory')
        return cv2.imread(image_path)
'''
#【Xavierポーティング】

def imwrite(save_path, image, params=None):
 
    logging.error('imwrite call')
    logging.error('save_path::%s',save_path)

    ext = osp.splitext(osp.basename(save_path))[-1]
    ret,encode_image = cv2.imencode(ext, image, params)
    if ret:
        with open(save_path, mode='w+b') as f:
            encode_image.tofile(f)

def calc_box_size(img, bias=0):
    common_size =  512
    height, width = img.shape[:2]
    if height > width:
        scale = common_size / height
    else:
        scale = common_size / width

    int_scale = int(Decimal(1/scale).quantize(Decimal('1'), ROUND_HALF_UP))
    fontScale = 0.5/scale + bias if int_scale else 0.5
    thickness = int_scale if int_scale else 1
    box_size = 2**int_scale if int_scale < 4 else 2**3+int_scale

    return fontScale, thickness, box_size
    
def write_bbox(img, bbox, label_name, color, score=None, mode="xy"):
    fontScale, thickness, box_size = calc_box_size(img)
    x1, y1, x2, y2 = [int(i) for i in bbox.tolist()]
    if mode == "wh":
        x2, y2 = x1+x2, y1+y2
    test = f"{score:.2f}:{label_name}" if score is not None else f"{label_name}"

    text_size = cv2.getTextSize(test, cv2.FONT_HERSHEY_SIMPLEX, fontScale, thickness)[0]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, box_size)
    cv2.rectangle(img, (x1, y1-text_size[1]), (x1+text_size[0]+3, y1), color, -1)
    cv2.putText(img, test, (x1, y1), 
        cv2.FONT_HERSHEY_SIMPLEX, fontScale, (0, 0, 0), thickness)
    return img

def create_sharedmemory(name, size):
#【Xavierポーティング】
    logging.error('create_sharedmemory call')
    logging.error('name::%s',name)
    print("create_sharedmemory call")

    fd = shm_open(name)

    ftruncate(fd, size)

    m = mmap.mmap(fd, size)

    shm_close(fd)

    return m
    '''
    # File descripter zero is shared memory
    m = mmap.mmap(0, size, name, access=mmap.ACCESS_WRITE)
    return m
    '''

def delete_sharedmemory(name, m):
    print("delete_sharedmemory call")
    m.close()
    shm_unlink(name)
#【Xavierポーティング】