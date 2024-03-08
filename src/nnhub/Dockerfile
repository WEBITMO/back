FROM python:3.9-slim

WORKDIR /opt/app-root/src/

RUN pip install install torch torchvision torchaudio

COPY src/nnhub/requirements.txt /opt/app-root/src/nnhub/
RUN pip install --no-cache-dir -r /opt/app-root/src/nnhub/requirements.txt

RUN apt-get update && apt-get install -y ffmpeg

ENV PYTHONPATH /opt/app-root/src
ENV PYTHONUNBUFFERED=1

CMD ["gunicorn", "--config", "/opt/app-root/src/nnhub/api/config/gunicorn_config.py"]