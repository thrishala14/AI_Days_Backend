from utils.env_config import EnvConfig
import os

async def startup_task():
    config = EnvConfig()
    os.environ['OPENAI_API_KEY'] = config.openai_api_key
