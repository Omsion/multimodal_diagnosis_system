
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from app.config.settings import settings, setup_logging
from app.api.routes import router
from app.services.diagnosis import diagnosis_service

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management"""
    logger.info("Starting Multimodal DR Diagnosis System...")
    
    # Initialize models
    diagnosis_service.initialize()
    
    yield
    logger.info("System shutting down")

app = FastAPI(
    title=settings.APP_NAME,
    description="Multimodal Diabetic Retinopathy Diagnosis System",
    version=settings.VERSION,
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.HOST, port=settings.PORT)
