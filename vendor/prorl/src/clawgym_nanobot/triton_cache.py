"""Triton cache writer for per-process directories on shared filesystems.

Avoid Triton's cross-directory rename followed by recursive removedirs: some
shared filesystems briefly fail to expose the renamed destination. A private
process directory permits direct, fsynced writes with read-back verification.
"""
import os,threading,time
from triton.runtime.cache import FileCacheManager
_LOCK=threading.RLock()
class PrivateProcessCacheManager(FileCacheManager):
 def put(self,data,filename,binary=True):
  payload=data if isinstance(data,bytes) else str(data).encode()
  path=self._make_path(filename)
  with _LOCK:
   os.makedirs(self.cache_dir,exist_ok=True)
   with open(path,'wb') as f:
    f.write(payload);f.flush();os.fsync(f.fileno())
   deadline=time.monotonic()+10
   while True:
    try:
     with open(path,'rb') as f:actual=f.read()
     if actual==payload:return path
    except FileNotFoundError:pass
    if time.monotonic()>=deadline:raise OSError('Triton cache read-back failed: '+path)
    time.sleep(.05)
 def get_file(self,filename):
  with _LOCK:return super().get_file(filename)
 def get_group(self,filename):
  with _LOCK:return super().get_group(filename)
