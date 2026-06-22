from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Config(BaseSettings):
    OPENAI_API_KEY: str = Field(init=False)
    GROQ_API_KEY: str = Field(init=False)
    GOOGLE_API_KEY: str = Field(init=False)

    model_config = SettingsConfigDict(env_file=".env")


config = Config()
