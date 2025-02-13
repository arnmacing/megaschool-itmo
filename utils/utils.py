import re

from utils.logger import setup_logger


def sanitize_user_input(text: str) -> str:
    """
    Блокируем прямые указания «Ignore all instructions», «system message» и т.д.,
    чтобы избежать промт-инъекций.
    """
    forbidden_patterns = [
        r"(?i)ignore all instructions",
        r"(?i)forget all instructions",
        r"(?i)system message",
    ]
    for pattern in forbidden_patterns:
        text = re.sub(pattern, "", text)
    return text


def clean_query(query: str) -> str:
    """
    Очищает запрос от мусора, но сохраняет нумерованные варианты ответа.
    """
    query = query.strip()
    query = re.sub(r"[^\w\s\d\.\,\?\!]", "", query)
    query = re.sub(r"\s+", " ", query)

    return query


def validate_urls(urls):
    return [u for u in urls if u.startswith("http")]


def fix_broken_urls(urls):
    fixed = []
    for url in urls:
        clean_url = re.sub(r"[\"',)]+$", "", url)
        fixed.append(clean_url)
    return fixed


async def call_openai_with_retry(request_function, *args, **kwargs):
    """Вызывает OpenAI с повторными попытками при сбоях."""
    logger = await setup_logger()
    max_retries = 3
    for attempt in range(max_retries):
        try:
            return await request_function(*args, **kwargs)
        except OSError as e:
            if "I/O operation on closed file" in str(e):
                await logger.error("Ошибка ввода-вывода: попытка повторного запроса.")
                continue
            raise
        except Exception as e:
            await logger.error(f"Ошибка OpenAI API: {str(e)}")
            raise
