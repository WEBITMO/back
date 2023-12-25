import random
from typing import Dict

from aiosqlite import Connection
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from pydantic import BaseModel
from starlette.status import HTTP_201_CREATED, HTTP_404_NOT_FOUND

from nnhub.api.schema.model import Model
from nnhub.domain.model import get_available_local_models
from nnhub.domain.types import ModelType
from nnhub.infrastructure.db import get_db, update_model_name
from nnhub.infrastructure.model_settings import model_settings

router = APIRouter(
    prefix='/image_classification_model',
    tags=['Image Classification Model'],
)


@router.get("/")
async def get_available_models() -> list[str]:
    return get_available_local_models(model_settings.IMAGE_CLASSIFICATION)


@router.post("/load")
async def load_model_locally(
    model: Model,
    db: Connection = Depends(get_db)
):
    available_models = get_available_local_models(model_settings.IMAGE_CLASSIFICATION)
    if model.name not in available_models:
        return HTTP_404_NOT_FOUND
    await update_model_name(db, model_type=ModelType.IMAGE_CLASSIFICATION, model_name=model.name)
    return HTTP_201_CREATED


class ImagePredictionResult(BaseModel):
    filename: str
    filesize: int
    prediction: Dict[str, float]


@router.post('/predict')
async def inference_by_image(
    image: UploadFile = File(..., media_type='image/*'),
    db: Connection = Depends(get_db),
) -> ImagePredictionResult:
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid image: type")

    current_model = await db.execute("SELECT name FROM model WHERE type = ?",
                                     (ModelType.IMAGE_CLASSIFICATION.value,))
    current_model = await current_model.fetchone()

    if current_model is None:
        raise HTTPException(status_code=404, detail="Model not found")
    current_model = current_model[0]

    if current_model == 'mnist':
        classes = 10
    elif current_model == 'cifar_100':
        classes = 100
    else:
        raise ValueError("Invalid model name")

    prediction = {str(i): random.random() for i in range(classes)}

    content = await image.read()

    return ImagePredictionResult(
        filename=image.filename,
        filesize=len(content),
        prediction=prediction
    )
