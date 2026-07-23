"""App settings. Loads the git-ignored .env into the process environment so
the provider and tool classes (which read os.environ) see it too."""

import os
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")


def load_env_file(path: str = ENV_PATH):
    """Loads KEY=value lines into os.environ without overwriting existing vars."""
    if not os.path.exists(path):
        return
    with open(path) as env_file:
        for line in env_file:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, _, value = line.partition("=")
            name, value = name.strip(), value.strip().strip("'\"")
            if name and name not in os.environ:
                os.environ[name] = value


class Settings(BaseSettings):
    chatboton_provider: str = Field(default="ollama")
    ollama_model: str = Field(default="chatboton-heretic")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    load_env_file()
    return Settings()
