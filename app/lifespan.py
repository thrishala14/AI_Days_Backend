"""
lifespan takes care of on startup and shutdown events.
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.startup import startup_task
from custom_exceptions.startup_error import StartupError
from utils.logger import logger

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    startup task is triggered in lifespan.
    """
    try:
        await startup_task()
    except Exception as e:
        logger.error("An error occurred in startup task: %s", str(e))
        raise StartupError(f"An error occurred in startup task: {e}")
    
    logger.info("✅ Server started successfully.")
    yield
    logger.info("🛑 Server stopped successfully.")
