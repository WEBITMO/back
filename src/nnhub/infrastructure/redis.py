import os

import aioredis

REDIS_URL = os.getenv("REDIS_URL")


async def get_redis() -> aioredis.Redis:
    redis = await aioredis.create_redis_pool(REDIS_URL)
    try:
        yield redis
    finally:
        redis.close()
        await redis.wait_closed()
