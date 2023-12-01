TODO (will change, be simplified):

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

export MODEL_SETTINGS_PATH_BASE=./model
export DB_SETTINGS_PATH=./db.sqlite

run python -m gunicorn nnhub.api.main:app --workers 1 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8080 --reload --config nnhub/infrastructure/gunicorn.conf.py
```
