```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install torch torchvision torchaudio

export MODEL_SETTINGS_PATH_BASE=./models
export DB_SETTINGS_PATH=./data/db.sqlite

python -m gunicorn --config src/nnhub/api/config/gunicorn_config.py
```

OR

```
docker compose -f docker-compose.yaml -p back up -d --build
```