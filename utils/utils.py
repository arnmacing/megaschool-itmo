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


import re

def clean_query(query: str) -> str:
    """
    Очищает запрос от мусора, но сохраняет нумерованные варианты ответа.
    """
    query = query.strip()
    query = re.sub(r"[^\w\s\d\.\,\?\!]", "", query)
    query = re.sub(r"\s+", " ", query)

    return query


def remove_links_from_reasoning(reasoning: str) -> str:
    """
    Удаляет http/https ссылки из reasoning, чтобы они не просачивались
    туда по требованиям.
    """
    return re.sub(r"https?://[^\s\"',)]+", "", reasoning).strip()


def extract_answer_from_reasoning(text: str):
    """
    Извлекает номер ответа (1-10), если встречается в формате "вариант X" или "вариант№ X".
    """
    clean_text = re.sub(r'[\"\',*]+', "", text.lower())
    clean_text = re.sub(r"[\n\r\t]+", " ", clean_text)
    clean_text = re.sub(r"\s+", " ", clean_text)

    pattern = r"вариант\w*\s*(\d+)"
    match = re.search(pattern, clean_text)
    if match:
        try:
            num = int(match.group(1))
            if 1 <= num <= 10:
                return num
        except ValueError:
            pass
    return None


def find_urls_in_text(text: str):
    url_pattern = re.compile(r"https?://[^\s\"',)]+")
    return url_pattern.findall(text)


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
