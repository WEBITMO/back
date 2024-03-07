from aiosqlite import Connection
from fastapi import APIRouter, Depends
from starlette.status import HTTP_201_CREATED, HTTP_404_NOT_FOUND

from nnhub.api.schema.model import Model
from nnhub.domain.model import get_available_local_models
from nnhub.domain.types import ModelType
from nnhub.infrastructure.db import get_db, update_model_name
from nnhub.infrastructure.model_settings import model_settings

router = APIRouter(
    prefix='/large_language_model',
    tags=['Large Language Model'],
)


@router.get("/")
async def get_available_models() -> list[str]:
    return get_available_local_models(model_settings.LARGE_LANGUAGE)


@router.post("/load")
async def load_model_locally(
    model: Model,
    db: Connection = Depends(get_db)
):
    available_models = get_available_local_models(model_settings.LARGE_LANGUAGE)
    if model.name not in available_models:
        return HTTP_404_NOT_FOUND
    await update_model_name(db, model_type=ModelType.LARGE_LANGUAGE, model_name=model.name)
    return HTTP_201_CREATED
