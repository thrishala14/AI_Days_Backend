from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import insight_trace
from app.lifespan import lifespan

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(insight_trace.router)
