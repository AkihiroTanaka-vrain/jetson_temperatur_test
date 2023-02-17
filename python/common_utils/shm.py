# pythonからCのshm_open/shm_unlinkを呼び出す
# 下記のソースコードを参考にした
# https://gist.github.com/jakirkham/100a7f5e86b0ff2a22de0850723a4c5c/c3fe5be93188022e0251c9d941bd500ecea09d90


import ctypes
import ctypes.util
import os
import stat

#libcだとリンクエラーになる。下記のサイトを参考にlibrtにする
#http://rbintelligence.blog.shinobi.jp/python/%E5%85%B1%E6%9C%89%E3%83%A1%E3%83%A2%E3%83%AA%E3%81%AB%E3%81%A4%E3%81%84%E3%81%A6
try:
    rt = ctypes.CDLL('librt.so')
except:
    rt = ctypes.CDLL('librt.so.1')

#libc = ctypes.cdll.LoadLibrary(ctypes.util.find_library("c"))

def shm_open(name):

    #型変換
    enc_name = name.encode('utf-8')
    p_name = ctypes.create_string_buffer(enc_name)

    result = rt.shm_open(
        p_name,
        ctypes.c_int(os.O_RDWR | os.O_CREAT),
        ctypes.c_ushort(stat.S_IRUSR | stat.S_IWUSR)
    )

    if result == -1:
        raise RuntimeError(os.strerror(ctypes.get_errno()))

    return result

def shm_close(fd):

    rt.close(fd)

# 0 :成功
# -1:エラー発生
def ftruncate(fd, length):

    fd = ctypes.c_int(fd)
    length = ctypes.c_long(length)
    result = rt.ftruncate(fd, length)

    if result == -1:
        raise RuntimeError(os.strerror(ctypes.get_errno()))

def shm_unlink(name):

    #型変換
    enc_name = name.encode('utf-8')
    p_name = ctypes.create_string_buffer(enc_name)

    result = rt.shm_unlink(p_name)

    if result == -1:
        raise RuntimeError(os.strerror(ctypes.get_errno()))