# utils.py
from loguru import logger
import os

def setup_logging():
    logger.remove()
    logger.add(lambda msg: print(msg, end=''), level='INFO')

def get_db_url(cli_db=None):
    env = os.getenv('DATABASE_URL')
    return cli_db or env
