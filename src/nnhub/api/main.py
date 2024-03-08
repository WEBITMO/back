import json
import os
from enum import Enum
from typing import Optional

import aiofiles
import aiohttp
import aioredis
import huggingface_hub
from fastapi import FastAPI, APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.concurrency import run_in_threadpool
from huggingface_hub import HfFileSystem, snapshot_download
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import RedirectResponse, StreamingResponse

from nnhub.api.image_classification_model.router import router as image_classification_router
from nnhub.api.image_segmentation_model.router import router as image_segmentation_router
from nnhub.api.speech_to_text_model.router import router as speech_to_text_model_router
from nnhub.infrastructure.model_settings import model_settings
from nnhub.infrastructure.redis import get_redis

fs = HfFileSystem()


app = FastAPI(
    title='Neural Networks Hub',
    summary='summary',
    description='description',
    version='0.0.1',
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
    if request.pipeline == PipelineTag.text_generation:
        async with aiofiles.open('/opt/app-root/data/models.json', mode='r') as file:
            return json.loads(await file.read())

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
                    model['available'] = model['pipeline_tag'] in [PipelineTag.image_classification, PipelineTag.object_detection, PipelineTag.automatic_speech_recognition] and model.get('gated', True) is False
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
async def get_model_card(org_id: str, repo_id: str, redis: aioredis.Redis = Depends(get_redis)):
    cache_key = f"readme_location:{org_id}:{repo_id}"

    cached_readme_location = await redis.get(cache_key)
    if cached_readme_location:
        readme_location = cached_readme_location.decode('utf-8')
    else:
        exist = huggingface_hub.file_exists(f'{org_id}/{repo_id}', filename='README.md')
        if not exist:
            raise HTTPException(status_code=404, detail="README.md not found")

        readme_location = huggingface_hub.hf_hub_download(f'{org_id}/{repo_id}', filename='README.md')
        await redis.set(cache_key, readme_location)

    async def file_generator():
        async with aiofiles.open(readme_location, mode='r') as file:
            yield await file.read()

    return StreamingResponse(file_generator(), media_type="text/markdown")


@router_v1.get("/model_remote_size/{org_id}/{repo_id}")
async def get_remote_model_size(org_id: str, repo_id: str, redis: aioredis.Redis = Depends(get_redis)):
    cache_key = f"model_size:{org_id}:{repo_id}"

    cached_value = await redis.get(cache_key)
    if cached_value is not None:
        return int(cached_value)

    infos = fs.ls(f'{org_id}/{repo_id}', recursive=True)
    total_size = 0
    for info in infos:
        if info['type'] == 'file':
            total_size += info['size']

    await redis.set(cache_key, total_size)

    return total_size


@router_v1.get("/model_local_size/{org_id}/{repo_id}")
async def get_local_model_size(org_id: str, repo_id: str):
    base_path = model_settings.BASE / org_id / repo_id
    total_size = 0

    for dirpath, dirnames, filenames in os.walk(base_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.isfile(fp):
                total_size += os.path.getsize(fp)

    return total_size


async def download_task(org_id: str, repo_id: str, download_key, redis: aioredis.Redis):
    print('download_task start')
    await run_in_threadpool(
        snapshot_download,
        f'{org_id}/{repo_id}',
        local_dir=model_settings.BASE / org_id / repo_id,
        force_download=True,
        local_dir_use_symlinks=False,
        max_workers=2,
    )
    print('download_task end')
    await redis.delete(download_key)


@router_v1.get("/model_download/{org_id}/{repo_id}")
async def model_download(org_id: str, repo_id: str, background_tasks: BackgroundTasks, redis: aioredis.Redis = Depends(get_redis)):
    download_key = f"download:{org_id}:{repo_id}"

    in_progress = await redis.get(download_key)
    if in_progress is not None:
        return "download in progress or completed"

    remote_model_size = await get_remote_model_size(org_id, repo_id, redis)
    local_model_size = await get_local_model_size(org_id, repo_id)

    if local_model_size == remote_model_size:
        return "model is up-to-date"

    await redis.set(download_key, "true")

    background_tasks.add_task(download_task, org_id, repo_id, download_key, redis)

    return "download started"


@router_v1.get("/model_download_status/{org_id}/{repo_id}")
async def model_download_status(org_id: str, repo_id: str, redis: aioredis.Redis = Depends(get_redis)):
    download_key = f"download:{org_id}:{repo_id}"
    download_status = await redis.get(download_key)

    if download_status is None:
        return {"status": "not_started"}
    else:
        return {"status": "in_progress"}


@router_v1.get("/model_load_status/{org_id}/{repo_id}")
async def model_load_status(org_id: str, repo_id: str, redis: aioredis.Redis = Depends(get_redis)):
    load_key = f"load:{org_id}:{repo_id}"
    load_status = await redis.get(load_key)

    if load_status is None:
        return {"status": "not_loaded"}
    else:
        return {"status": "loaded"}


router_v1.include_router(image_classification_router)
router_v1.include_router(image_segmentation_router)
router_v1.include_router(speech_to_text_model_router)
app.include_router(router_v1)
