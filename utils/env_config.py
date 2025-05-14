from pydantic_settings import BaseSettings
import base64

class EnvConfig(BaseSettings):
    OPENAI_API_KEY_ENCODED: str
    host: str = "127.0.0.1"
    port: int = 8000

    class Config:
        env_file = ".env"
        extra = "allow"

    @property
    def openai_api_key(self) -> str:
        return base64.b64decode(self.OPENAI_API_KEY_ENCODED).decode()

envconfig = EnvConfig()


