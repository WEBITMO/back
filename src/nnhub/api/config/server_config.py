from pydantic_settings import BaseSettings, SettingsConfigDict


class ServerConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix='CONFIG_SERVER_')

    HOST: str = '0.0.0.0'
    PORT: int = 8888
    WORKERS_COUNT: int = 1
    AUTO_RELOAD: bool = True
    TIMEOUT: int = 60 * 60 * 24


server_config = ServerConfig()
