from os import kill
from signal import SIGINT

from gunicorn.workers.base import Worker

from nnhub.api.config.server_config import server_config

wsgi_app = "nnhub.api.main:app"
worker_class = "uvicorn.workers.UvicornWorker"
bind = f"{server_config.HOST}:{server_config.PORT}"
workers = server_config.WORKERS_COUNT
reload = server_config.AUTO_RELOAD
timeout = server_config.TIMEOUT


def worker_int(worker: Worker) -> None:
    kill(worker.pid, SIGINT)
