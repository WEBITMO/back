from contextlib import asynccontextmanager

from fastapi import FastAPI, APIRouter
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import RedirectResponse

from nnhub.api.image_classification_model.router import router as image_classification_router
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
router_v1.include_router(image_classification_router)
router_v1.include_router(large_language_model_router)
router_v1.include_router(speech_to_text_model_router)
app.include_router(router_v1)
