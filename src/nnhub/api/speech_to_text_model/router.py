import aioredis
import lorem
from aiosqlite import Connection
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from pydantic import BaseModel
from starlette.status import HTTP_201_CREATED, HTTP_404_NOT_FOUND, HTTP_200_OK
from transformers import pipeline

from nnhub.domain.types import ModelType
from nnhub.infrastructure.model_settings import model_settings
from nnhub.infrastructure.redis import get_redis

router = APIRouter(
    prefix='/automatic-speech-recognition',
    tags=['Speech-To-Text Model'],
)


@router.get("/load/{org_id}/{repo_id}")
async def load_model_locally(
    org_id: str,
    repo_id: str,
    redis: aioredis.Redis = Depends(get_redis),
):
    load_key = f"load:{org_id}:{repo_id}"
    load_status = await redis.get(load_key)
    if load_status:
        return HTTP_201_CREATED

    global pipe

    pipe = pipeline("automatic-speech-recognition", model=model_settings.BASE / org_id / repo_id)
    await redis.set(load_key, "true")

    return HTTP_201_CREATED


@router.get("/unload/{org_id}/{repo_id}")
async def unload_model_locally(
    org_id: str,
    repo_id: str,
    redis: aioredis.Redis = Depends(get_redis),
):
    load_key = f"load:{org_id}:{repo_id}"
    load_status = await redis.get(load_key)
    if not load_status:
        return HTTP_404_NOT_FOUND
    await redis.delete(load_key)
    global pipe
    pipe = None
    return HTTP_200_OK


class AudioPredictionResult(BaseModel):
    transcribed_text: str


@router.post('/predict')
async def inference_by_audio(
    audio: UploadFile = File(..., media_type='audio/*'),
) -> AudioPredictionResult:
    global pipe
    if not audio.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="Invalid audio type")

    content = await audio.read()

    prediction = pipe(content).get("text", None)

    return AudioPredictionResult(
        transcribed_text=prediction or lorem.text()
    )
