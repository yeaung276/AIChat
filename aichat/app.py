import os
import yaml
import logging
from contextlib import asynccontextmanager
from dotenv import load_dotenv

from pyngrok import ngrok

from sqlmodel import SQLModel
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from aichat.db_models.db import engine
from aichat.pipeline.factory import ModelFactory
from aichat.routes import chatRouter, userRouter, adminRouter

logger = logging.getLogger("uvicorn")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # loading env
    logger.info("loading env file...")
    load_dotenv()
    
    # Initialize DB
    logger.info("setting up database...")
    SQLModel.metadata.create_all(engine)
    
    # Initialize Models
    config_f = os.getenv("CONFIG_FILE", "config.yaml")
    logger.info("Loading models using configuration from %s. Use CONFIG_FILE to change this.", config_f)
    with open(config_f, "r") as f:
        config = yaml.safe_load(f)

    ModelFactory.configure(config)
    
    # Setting up tunneling
    tunnel = None
    if os.getenv("NGROK_API_KEY"):
        logger.info("NGROK_API_KEY found. Setting up ngrok tunneling...")
        ngrok.set_auth_token(os.getenv("NGROK_API_KEY", ""))
        tunnel = ngrok.connect(
            addr=os.getenv("NGROK_APP_PORT"),
        )
        logger.info("APP available at %s", tunnel.public_url)
    yield
    if tunnel:
        ngrok.disconnect(tunnel.public_url) # type: ignore

app = FastAPI(lifespan=lifespan)

app.include_router(chatRouter)
app.include_router(userRouter)
app.include_router(adminRouter)
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

@app.get("/feedback")
async def feedback_page():
    return FileResponse("frontend/feedback.html")

