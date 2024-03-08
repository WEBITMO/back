```
python3 -m venv venv
source venv/bin/activate
pip install -r src/nnhub/requirements.txt
pip install torch torchvision torchaudio

export PYTHONPATH=./src
export MODEL_SETTINGS_PATH_BASE=./models
export DB_SETTINGS_PATH=./data/db.sqlite
export REDIS_URL=redis://:redis_password@localhost:6379/0

python -m gunicorn --config src/nnhub/api/config/gunicorn_config.py
```

OR

```
docker compose -f docker-compose.yaml -p back up -d --build
```