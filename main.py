import os
import yaml
import logging
from contextlib import asynccontextmanager

from sqlmodel import SQLModel
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from aichat.db_models.db import engine
from aichat.pipeline.factory import ModelFactory
from aichat.routes import chatRouter, userRouter
from aichat.security.auth import get_current_user

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize DB
    SQLModel.metadata.create_all(engine)
    
    # Initialize Models
    config_f = os.getenv("CONFIG_FILE", "config.yaml")
    logger.info("Using configuration from %s. Use CONFIG_FILE to change this.", config_f)
    with open(config_f, "r") as f:
        config = yaml.safe_load(f)

    await ModelFactory.configure(config)
    yield

app = FastAPI(lifespan=lifespan)

app.include_router(chatRouter)
app.include_router(userRouter)
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")


@app.get("/")
async def index():
    return FileResponse("frontend/index.html")

@app.get("/login")
async def login_page():
    return FileResponse("frontend/login.html")

@app.get("/register")
async def register_page():
    return FileResponse("frontend/register.html")

