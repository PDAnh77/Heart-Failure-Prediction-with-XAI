from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    DATABASE_URL: str
    DATABASE_KEY: str
    SECRET_JWT_KEY: str
    RENDER_APP_URL: str
    NEXT_APP_URL: str

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        secrets_dir="/run/secrets",
        extra="ignore"
    )

settings = Settings()