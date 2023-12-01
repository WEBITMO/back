from contextlib import asynccontextmanager

import aiosqlite
from aiosqlite import Connection, Cursor

from nnhub.domain.types import ModelType
from nnhub.infrastructure.db_settings import db_settings


async def create_db():
    return await aiosqlite.connect(db_settings.PATH)


async def get_db():
    async with aiosqlite.connect(db_settings.PATH) as db:
        yield db


async def init_db(db: Connection):
    await db.execute("""
                        CREATE TABLE IF NOT EXISTS model (
                            id INTEGER PRIMARY KEY,
                            type TEXT NOT NULL,
                            name TEXT
                        )
                        """)
    await db.commit()


async def update_model_name(db: Connection, model_type: ModelType, model_name: str):
    cursor = await db.execute("UPDATE model SET name = ? WHERE type = ?",
                              (model_name, model_type.value))
    if cursor.rowcount == 0:
        await db.execute("INSERT INTO model (type, name) VALUES (?, ?)",
                         (model_type.value, model_name))
    await db.commit()
