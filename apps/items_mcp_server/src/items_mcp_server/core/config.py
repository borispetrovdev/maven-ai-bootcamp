from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    OPENAI_API_KEY: str = Field(init=False)
    CO_API_KEY: str = Field(init=False)

    model_config = SettingsConfigDict(env_file=".env")


config = Config()
