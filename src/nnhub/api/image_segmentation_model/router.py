import io

import aioredis
import torch
from PIL import Image
from aiosqlite import Connection
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from pydantic import BaseModel
from starlette.status import HTTP_201_CREATED, HTTP_404_NOT_FOUND, HTTP_200_OK
from transformers import pipeline, Pipeline

from nnhub.infrastructure.db import get_db
from nnhub.infrastructure.model_settings import model_settings
from nnhub.infrastructure.redis import get_redis

router = APIRouter(
    prefix='/object-detection',
    tags=['Image Segmentation Model'],
)

pipe: Pipeline = None


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

    pipe = pipeline("object-detection", model=model_settings.BASE / org_id / repo_id)
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


class ImageSegmentationResult(BaseModel):
    label: str
    score: float
    box: tuple[int, int, int, int]


class ImageSegmentationResults(BaseModel):
    predictions: list[ImageSegmentationResult]


@router.post('/predict')
async def inference_by_image(
    image: UploadFile = File(..., media_type='image/*'),
    db: Connection = Depends(get_db),
) -> ImageSegmentationResults:
    global pipe
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid image: type")

    content = await image.read()

    pil_image = Image.open(io.BytesIO(content)).convert('RGB')

    predictions = pipe(pil_image)
    result = []

    for prediction in predictions:
        label = prediction['label']
        score = round(prediction['score'], 3)
        box_dict = prediction['box']
        xmin, ymin, xmax, ymax = box_dict['xmin'], box_dict['ymin'], box_dict['xmax'], box_dict['ymax']
        box = (xmin, ymin, xmax - xmin, ymax - ymin)
        result.append(ImageSegmentationResult(label=label, score=score, box=box))

    return ImageSegmentationResults(predictions=result)
