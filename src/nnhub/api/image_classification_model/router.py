import io

import aioredis
from PIL import Image
from aiosqlite import Connection
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from pydantic import BaseModel
from starlette.status import HTTP_201_CREATED, HTTP_404_NOT_FOUND, HTTP_200_OK
from transformers import pipeline, Pipeline

from nnhub.infrastructure.model_settings import model_settings
from nnhub.infrastructure.redis import get_redis

router = APIRouter(
    prefix='/image-classification',
    tags=['Image Classification Model'],
)

pipe: Pipeline = None


# /@router.get("/")
# async def get_available_models() -> list[str]:
#     return get_available_local_models(model_settings.IMAGE_CLASSIFICATION)


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

    pipe = pipeline("image-classification", model=model_settings.BASE / org_id / repo_id)
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


class ImagePredictionResult(BaseModel):
    predicted_class: str


@router.post('/predict')
async def inference_by_image(
    image: UploadFile = File(..., media_type='image/*'),
    redis: aioredis.Redis = Depends(get_redis),
) -> ImagePredictionResult:
    global pipe
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid image: type")

    content = await image.read()

    pil_image = Image.open(io.BytesIO(content)).convert('RGB')

    prediction = pipe(pil_image)

    predicted_class = prediction[0]["label"]

    return ImagePredictionResult(predicted_class=predicted_class)
