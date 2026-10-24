"""Cooperate with the existing English GPU mutex; never stop its processes."""
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import subprocess
import time

LOCK = Path("/tmp/phone69_460h_serial.lock")


def gpu_processes():
    return subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                                   "--format=csv,noheader,nounits"], text=True).strip()


@contextmanager
def gpu_lease(*, wait=False):
    with LOCK.open("a") as stream:
        while True:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                if not wait:
                    raise RuntimeError("English/Chinese queue holds the GPU mutex; no task was interrupted")
            else:
                if not gpu_processes():
                    break
                fcntl.flock(stream, fcntl.LOCK_UN)
                if not wait:
                    raise RuntimeError("GPU has live processes; no task was interrupted")
            time.sleep(30)
        yield stream.fileno()


def verify_inherited_lease(fd):
    if fd is None or os.fstat(fd).st_ino != LOCK.stat().st_ino:
        raise ValueError("Run through the Chinese pipeline holding the shared English GPU mutex")
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
