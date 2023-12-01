import os
import signal


def worker_int(worker):
    # for hot reload
    os.kill(worker.pid, signal.SIGINT)
