import os
from pathlib import Path

from nnhub.infrastructure.model_settings import model_settings


def get_available_local_models(model_dir: Path) -> list[str]:
    return os.listdir(model_settings.BASE / model_dir)
