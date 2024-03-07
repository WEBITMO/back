from aiosqlite import Connection
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from pydantic import BaseModel
from starlette.status import HTTP_201_CREATED, HTTP_404_NOT_FOUND
from transformers import pipeline

from nnhub.api.schema.model import Model
from nnhub.domain.model import get_available_local_models
from nnhub.domain.types import ModelType
from nnhub.infrastructure.db import get_db, update_model_name
from nnhub.infrastructure.model_settings import model_settings
import lorem

router = APIRouter(
    prefix='/speech_to_text_model',
    tags=['Speech-To-Text Model'],
)


@router.get("/")
async def get_available_models() -> list[str]:
    return get_available_local_models(model_settings.SPEECH_TO_TEXT)


@router.post("/load")
async def load_model_locally(
    model: Model,
    db: Connection = Depends(get_db)
):
    global pipe
    available_models = get_available_local_models(model_settings.SPEECH_TO_TEXT)
    if model.name not in available_models:
        return HTTP_404_NOT_FOUND
    await update_model_name(db, model_type=ModelType.SPEECH_TO_TEXT, model_name=model.name)
    pipe = pipeline("automatic-speech-recognition", model=model_settings.BASE / model_settings.SPEECH_TO_TEXT / model.name)
    return HTTP_201_CREATED


class AudioPredictionResult(BaseModel):
    transcribed_text: str


@router.post('/predict')
async def inference_by_audio(
    audio: UploadFile = File(..., media_type='audio/*'),
    db: Connection = Depends(get_db),
) -> AudioPredictionResult:
    global pipe
    if not audio.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="Invalid audio type")

    current_model = await db.execute("SELECT name FROM model WHERE type = ?",
                                     (ModelType.SPEECH_TO_TEXT.value,))
    current_model = await current_model.fetchone()

    if current_model is None:
        raise HTTPException(status_code=404, detail="Model not found")
    current_model = current_model[0]

    content = await audio.read()

    prediction = pipe(content).get("text", None)

    return AudioPredictionResult(
        transcribed_text=prediction or lorem.text()
    )
