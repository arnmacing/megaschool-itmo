import os
import sys

from aiologger import Logger
from aiologger.formatters.base import Formatter
from aiologger.handlers.files import AsyncFileHandler
from aiologger.handlers.streams import AsyncStreamHandler
from aiologger.levels import LogLevel

_logger_instance = None


async def setup_logger():
    global _logger_instance

    if _logger_instance is not None:
        return _logger_instance

    logger = Logger(name="api_logger", level=LogLevel.INFO)

    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    formatter = Formatter(
        fmt="{asctime} | {levelname} | {message}",
        datefmt="%Y-%m-%d %H:%M:%S",
        style="{",
    )

    file_handler = AsyncFileHandler(filename="logs/api.log", mode="a", encoding="utf-8")
    file_handler.formatter = formatter

    stream_handler = AsyncStreamHandler(stream=sys.stdout)
    stream_handler.formatter = formatter

    logger.add_handler(file_handler)
    logger.add_handler(stream_handler)

    _logger_instance = logger
    return logger
