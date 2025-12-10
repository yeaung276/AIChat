import os
import yaml
import logging
import argparse

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from aichat.rtc import negotiator
from aichat.pipeline.factory import ModelFactory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.include_router(negotiator)
app.mount("/static", StaticFiles(directory="frontend"), name="static")


@app.get("/")
async def index():
    return FileResponse("frontend/index.html")


def setup(args):
    logger.info(f"setting up with configuration {args.config}")
    if not os.path.exists(args.config):
        raise FileNotFoundError("configuration file not found in the path.")
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    ModelFactory.configure(config)
    
def main(args):
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the server with config.")

    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to YAML configuration file."
    )

    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the server to. Default: 0.0.0.0"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to run the server on. Default: 8080"
    )

    args = parser.parse_args()

    setup(args)
    main(args)