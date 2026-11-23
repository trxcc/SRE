"""256 cross-node POSIX record-lock permits with process-local reservations."""
import asyncio
import errno
import fcntl
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path

ROOT = Path(os.environ['RESEARCHCLAW_JUDGE_SLOT_ROOT'])
CAP = 256
LOCK_KIND = 'posix_lockf_global256_v1'
_local_lock = threading.Lock()
_local_in_use = set()


def _after_fork():
    global _local_lock, _local_in_use
    _local_lock = threading.Lock()
    _local_in_use = set()


if hasattr(os, 'register_at_fork'):
    os.register_at_fork(after_in_child=_after_fork)


def _try_slot():
    # POSIX locks are process-owned: skip local reservations before opening the
    # file, and serialize threads so another coroutine cannot reenter a slot.
    with _local_lock:
        for index in range(CAP):
            if index in _local_in_use:
                continue
            fd = os.open(ROOT / f'slot-{index:02d}.lock',
                         os.O_RDWR | os.O_CREAT | getattr(os, 'O_NOFOLLOW', 0), 0o600)
            try:
                fcntl.lockf(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                os.close(fd)
                if exc.errno in (errno.EACCES, errno.EAGAIN):
                    continue
                raise
            except BaseException:
                os.close(fd)
                raise
            _local_in_use.add(index)
            return index, fd
    return None


@asynccontextmanager
async def request_slot():
    if not ROOT.is_dir() or ROOT.is_symlink():
        raise RuntimeError('judge256 slot root unavailable')
    held = None
    while held is None:
        held = _try_slot()
        if held is None:
            await asyncio.sleep(.05)
    index, fd = held
    try:
        yield
    finally:
        with _local_lock:
            try:
                fcntl.lockf(fd, fcntl.LOCK_UN)
            finally:
                try:
                    os.close(fd)
                finally:
                    _local_in_use.remove(index)


def install(provider):
    current = getattr(provider, '_researchclaw_global_request_limit', None)
    kind = getattr(provider, '_researchclaw_global_request_lock_kind', None)
    if current == CAP and kind == LOCK_KIND:
        return
    if current is not None:
        raise RuntimeError('judge256 mixed request-gate versions; fresh process required')
    for name in ('chat', 'chat_stream'):
        original = getattr(provider, name)

        def wrap(fn):
            async def limited(self, *args, **kwargs):
                async with request_slot():
                    return await fn(self, *args, **kwargs)
            limited.__name__ = fn.__name__
            return limited

        setattr(provider, name, wrap(original))
    provider._researchclaw_global_request_limit = CAP
    provider._researchclaw_global_request_lock_kind = LOCK_KIND
