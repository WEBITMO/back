from contextlib import asynccontextmanager
from enum import Enum
from typing import Optional

import aiofiles
import aiohttp
import huggingface_hub
from fastapi import FastAPI, APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import RedirectResponse, StreamingResponse

from nnhub.api.image_classification_model.router import router as image_classification_router
from nnhub.api.image_segmentation_model.router import router as image_segmentation_router
from nnhub.api.large_language_model.router import router as large_language_model_router
from nnhub.api.speech_to_text_model.router import router as speech_to_text_model_router
from nnhub.infrastructure.db import init_db, create_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    db = await create_db()
    await init_db(db)
    await db.close()
    yield


app = FastAPI(
    title='Neural Networks Hub',
    summary='summary',
    description='description',
    version='0.0.1',
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
async def redirect_from_root() -> RedirectResponse:
    return RedirectResponse(url='/docs')


router_v1 = APIRouter(
    prefix='/api/v1'
)


class Pipeline(BaseModel):
    id: str
    label: str
    sub_type: str = Field(alias="subType")


class PipelineTag(str, Enum):
    all = 'all'
    text_generation = 'text-generation'
    image_classification = 'image-classification'
    object_detection = 'object-detection'
    automatic_speech_recognition = 'automatic-speech-recognition'


@router_v1.get("/pipelines")
async def get_available_pipelines() -> list[Pipeline]:
    return [
        Pipeline(
            id=PipelineTag.text_generation,
            label='Text Generation',
            subType='nlp'
        ),
        Pipeline(
            id=PipelineTag.image_classification,
            label='Image Classification',
            subType='cv'
        ),
        Pipeline(
            id=PipelineTag.object_detection,
            label='Object Detection',
            subType='cv'
        ),
        Pipeline(
            id=PipelineTag.automatic_speech_recognition,
            label='Speech Recognition',
            subType='audio'
        ),
    ]


class Sort(str, Enum):
    trending = 'trending'
    likes = 'likes'
    downloads = 'downloads'
    created = 'created'
    updated = 'updated'


class AvailableModelsRequest(BaseModel):
    sort: Sort = Sort.trending
    pipeline: PipelineTag = PipelineTag.all
    p: Optional[int] = None
    search: Optional[str] = None


@router_v1.get("/models")
async def get_available_models(
    request: AvailableModelsRequest = Depends()
):
    params = {
        'library': 'transformers',
        'sort': request.sort.value,
    }
    if request.pipeline.value != PipelineTag.all:
        params['pipeline_tag'] = request.pipeline.value
    if request.p:
        params['p'] = request.p - 1
    if request.search:
        params['search'] = request.search
    async with aiohttp.ClientSession() as session:
        async with session.get(f'https://huggingface.co/models-json', params=params) as response:
            response = await response.json()
            try:
                models = response['models']
                for model in models:
                    if 'pipeline_tag' not in model.keys():
                        model['pipeline_tag'] = 'unknown'
                    model['available'] = model['pipeline_tag'] in [PipelineTag.image_classification, PipelineTag.object_detection, PipelineTag.automatic_speech_recognition]
                return response
            except KeyError as e:
                # print('ERROR')
                # print(e)
                # print(response)
                return response


async def read_file_async(file_path: str):
    async with aiofiles.open(file_path, mode='r') as file:
        return await file.read()


@router_v1.get("/model_card/{org_id}/{repo_id}")
async def get_model_card(org_id: str, repo_id: str):
    exist = huggingface_hub.file_exists(f'{org_id}/{repo_id}', filename='README.md')

    if not exist:
        raise HTTPException(status_code=404, detail="README.md not found")

    readme_location = huggingface_hub.hf_hub_download(f'{org_id}/{repo_id}', filename='README.md')

    async def file_generator():
        async with aiofiles.open(readme_location, mode='r') as file:
            yield await file.read()

    # background_tasks.add_task(os.remove, readme_location)

    return StreamingResponse(file_generator(), media_type="text/markdown")


router_v1.include_router(image_classification_router)
router_v1.include_router(image_segmentation_router)
router_v1.include_router(large_language_model_router)
router_v1.include_router(speech_to_text_model_router)
app.include_router(router_v1)
