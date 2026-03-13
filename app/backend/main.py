import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from inference import load_model, scan_area

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    token = os.getenv("MAPBOX_TOKEN", "")
    if not token:
        logger.error("MAPBOX_TOKEN is not set — tile fetching will fail. Check app/backend/.env")
    else:
        logger.info("MAPBOX_TOKEN loaded (%s...)", token[:8])
    load_model()
    yield


app = FastAPI(title="TrackFinder API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class ScanRequest(BaseModel):
    lat: float
    lng: float
    radius_km: float = Field(default=5.0, ge=1.0, le=15.0)
    threshold: float = Field(default=0.65, ge=0.5, le=0.99)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/scan")
async def scan(req: ScanRequest):
    return await scan_area(req.lat, req.lng, req.radius_km, req.threshold)
