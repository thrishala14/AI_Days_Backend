from utils.env_config import EnvConfig
import os

async def startup_task():
    config = EnvConfig()
    os.environ['OPENAI_API_KEY'] = config.openai_api_key
    # os.environ['LANGSMITH_TRACING'] = str(config.LANGSMITH_TRACING).lower()
    # os.environ['LANGSMITH_API_KEY'] = config.LANGSMITH_API_KEY
