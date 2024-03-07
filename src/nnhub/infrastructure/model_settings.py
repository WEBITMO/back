from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelPathSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix='MODEL_SETTINGS_PATH_')

    BASE: Path
    IMAGE_CLASSIFICATION: Path = 'image_classification_model'
    IMAGE_SEGMENTATION: Path = 'image_segmentation_model'
    LARGE_LANGUAGE: Path = 'large_language_model'
    SPEECH_TO_TEXT: Path = 'speech_to_text_model'


model_settings = ModelPathSettings()
