from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class DBSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix='DB_SETTINGS_')

    PATH: Path


db_settings = DBSettings()
