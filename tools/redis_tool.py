#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @ description:
import redis
import uuid
import time
import math
from configs import REDIS_HOST, REDIS_PASSWORD, REDIS_PORT, REDIS_TIMEOUT
from configs import logger


class RedisFactory(object):
    logger.info(f"redis connection info {REDIS_HOST}:{REDIS_PORT}")
    repo = dict()

    @classmethod
    def create(cls, db, decode_responses=False, timeout=None):
        try:
            pool = cls.repo[db]
        except KeyError:
            pool = redis.StrictRedis(host=REDIS_HOST,
                                     port=REDIS_PORT,
                                     password=REDIS_PASSWORD,
                                     socket_timeout=REDIS_TIMEOUT if timeout is None else timeout,
                                     db=db,
                                     decode_responses=decode_responses)

            cls.repo[db] = pool
        return cls.repo.get(db)

    @classmethod
    def get_or_create(cls, db, timeout=None):
        return cls.create(db, False, timeout)

    @classmethod
    def close(cls, db):
        try:
            cls.repo[db].disconnect()
            del cls.repo[db]
        except KeyError:
            pass

    @classmethod
    def close_all(cls):
        for db, pool in cls.repo.items():
            pool.disconnect()
        cls.repo = {}


def acquire_lock_with_timeout(conn, lock_name, acquire_timeout=3, lock_timeout=2):
    """
    基于 Redis 实现的分布式锁

    :param conn: Redis 连接
    :param lock_name: 锁的名称
    :param acquire_timeout: 获取锁的超时时间，默认 3 秒
    :param lock_timeout: 锁的超时时间，默认 2 秒
    :return:
    """

    identifier = str(uuid.uuid4())
    lock_timeout = int(math.ceil(lock_timeout))

    end = time.time() + acquire_timeout

    while time.time() < end:
        # 如果不存在这个锁则加锁并设置过期时间，避免死锁
        if conn.set(lock_name, identifier, ex=lock_timeout, nx=True):
            return identifier
        # 如果锁存在但是没有失效时间, 则进行设置, 避免出现死锁
        elif conn.ttl(lock_name) == -1:
            conn.expire(lock_name, lock_timeout)
        time.sleep(0.001)
    return False


def release_lock(conn, lock_name, identifier):
    """
    释放锁

    :param conn: Redis 连接
    :param lock_name: 锁的名称
    :param identifier: 锁的标识
    :return:
    """
    unlock_script = """
    if redis.call("get",KEYS[1]) == ARGV[1] then
        return redis.call("del",KEYS[1])
    else
        return 0
    end
    """
    unlock = conn.register_script(unlock_script)
    result = unlock(keys=[lock_name], args=[identifier])
    if result:
        return True
    else:
        return False

    