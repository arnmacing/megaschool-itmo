import os
import re
import string
import json
from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
from openai import AsyncOpenAI

from schemas.request import PredictionRequest, PredictionResponse
from tools.functions import search_links

load_dotenv()

app = FastAPI()

MODEL_NAME = "gpt-4"
client = None

current_year = datetime.now().year

system_message = {
    "role": "system",
    "content": (
        f"Ты — интеллектуальный агент, предоставляющий актуальную информацию об Университете ИТМО на {current_year} год.\n\n"
        "Твоя задача — **найти и дать точный ответ** (если есть варианты, указывай номер). Если информации недостаточно, **автоматически** используй поиск (`search_links`). Даже если ты уверен в ответе, **всегда используй функцию `search_links` для предоставления источников**.\n\n"
        "Формат итогового JSON (который формирует сервер, а не ты):\n"
        "{\n"
        '  "id": <число>,\n'
        '  "answer": <число или null>,\n'
        '  "reasoning": <строка>,\n'
        '  "sources": <список ссылок>\n'
        "}\n\n"
        "Как отвечать:\n"
        "1️⃣ **Всегда давай точный ответ** — не задавай встречных вопросов.\n"
        "2️⃣ **Если есть варианты (1..10)**, **обязательно указывай правильный номер** в reasoning.\n"
        "3️⃣ **Обязательно используй поиск** (`search_links`) для предоставления релевантных источников, даже если ты уверен в ответе.\n"
        "4️⃣ **Возвращай только reasoning** — не вставляй JSON, не выводи id, answer, sources напрямую.\n"
        "5️⃣ **Указывай модель** — в конце reasoning добавляй: `Ответ сгенерирован моделью GPT-4.`\n\n"
        "# Пример с вариантами:\n"
        'Вопрос: "Когда откроется приём документов в магистратуру?\\n1. 15 июня  2. 20 июня   3. 25 июня  4. 30 июня"\n'
        'Ответ reasoning: "Приём документов в магистратуру ИТМО начинается 20 июня. Это соответствует варианту 2. Ответ сгенерирован моделью GPT-4."\n\n'
        "# Ошибки, которых нужно избегать:\n"
        "- ❌ **Не игнорируй варианты ответов** (если есть 1..10, выбери правильный).\n"
        "- ❌ **Не вставляй ответ в JSON-формате в reasoning**\n"
        "- ❌ **Не задавай встречные вопросы.**\n"
        "- ❌ **Не вставляй ссылки и источник в reasoning.**\n\n"
        "Важно: Всегда предоставляй точную информацию, используя проверенные источники."
    ),
}


@app.on_event("startup")
async def startup_event():
    global client
    openai_api_key = os.getenv("OPENAI_API_KEY", "")
    if not openai_api_key:
        raise RuntimeError("OPENAI_API_KEY не задан в переменных окружения.")
    client = AsyncOpenAI(api_key=openai_api_key)


def clean_query(text: str) -> str:
    """
    Очищает текст запроса от лишних символов, спецсимволов, разрывов строк и пробелов.
    Удаляет все символы пунктуации, кроме точек, и удаляет точки после номеров вариантов.
    """
    # 1. Заменяем переносы строк и табуляции на пробелы
    text = re.sub(r"[\n\r\t]+", " ", text)

    # 2. Упрощаем пробелы
    text = re.sub(r"\s+", " ", text).strip()

    # 3. Удаляем нежелательные символы пунктуации (кроме точек)
    punctuation_to_remove = string.punctuation.replace(".", "")
    translator = str.maketrans('', '', punctuation_to_remove)
    text = text.translate(translator)

    # 4. Удаляем точки после номеров вариантов (1., 2., и т.д.)
    text = re.sub(r"\b(\d+)\.", r"\1", text)

    return text


def extract_answer_from_reasoning(text: str) -> Optional[int]:
    """
    Извлекает номер ответа (1-10) из reasoning, если он указан в формате 'вариант X' или 'option X'.
    """
    clean_text = re.sub(r"[\"',*]+", "", text.lower())
    clean_text = re.sub(r"[\n\r\t]+", " ", clean_text)  # Удаляем переносы строк и табуляции
    clean_text = re.sub(r"\s+", " ", clean_text)  # Замена множественных пробелов на один

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


def find_urls_in_text(text: str) -> List[str]:
    """
    Ищет ссылки (http/https) в тексте, возвращает список.
    """
    # Обновлённое регулярное выражение для извлечения чистых URL
    url_pattern = re.compile(r"https?://[^\s\"',)]+")
    return url_pattern.findall(text)


def validate_urls(urls: List[str]) -> List[str]:
    """
    Упрощённая проверка URL: возвращаем строки, начинающиеся с http.
    """
    return [u for u in urls if u.startswith("http")]


def fix_broken_urls(urls: List[str]) -> List[str]:
    """
    Убираем из ссылок лишние символы (кавычки, запятые, закрывающие скобки и т.д.).
    """
    fixed = []
    for url in urls:
        clean_url = re.sub(r"[\"',)]+$", "", url)
        fixed.append(clean_url)
    return fixed


async def predict(body: PredictionRequest) -> PredictionResponse:
    """
    Асинхронная функция обработки вопроса.
    Вызывает OpenAI API и search_links() при необходимости.
    """
    try:
        request_id = body.id
        query_text = clean_query(body.query)

        print("Очищенный запрос (query_text):")
        print(query_text)

        has_variants = bool(re.search(r"\b\d+\s", query_text))
        print("\nhas_variants:", has_variants)

        user_message = {"role": "user", "content": query_text}

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_links",
                    "description": "Search for ITMO-related links using serper.dev API",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"},
                            "max_results": {
                                "type": "number",
                                "description": "Number of results",
                            },
                        },
                        "required": ["query"],
                    },
                },
            }
        ]

        completion = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[system_message, user_message],
            tools=tools,
            tool_choice="auto",
        )
        message_content = completion.choices[0].message
        sources_list: List[str] = []

        if message_content.tool_calls:
            tool_call = message_content.tool_calls[0]
            tool_call_id = tool_call.id

            if tool_call.function.name == "search_links":
                fn_args = json.loads(tool_call.function.arguments)
                query_arg = fn_args.get("query", "")
                max_res_arg = fn_args.get("max_results", 3)

                print("Вызываем search_links")
                search_result = search_links(query_arg, max_res_arg)

                follow_up_messages = [
                    system_message,
                    user_message,
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": tool_call_id,
                                "type": "function",
                                "function": {
                                    "name": "search_links",
                                    "arguments": json.dumps(fn_args),
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "name": "search_links",
                        "content": json.dumps(search_result),
                    },
                ]

                second_completion = await client.chat.completions.create(
                    model=MODEL_NAME, messages=follow_up_messages
                )
                final_answer_text = second_completion.choices[0].message.content

                if "links" in search_result:
                    sources_list = [link["link"] for link in search_result["links"][:3]]
                else:
                    sources_list = []
            else:
                final_answer_text = (
                    "Функция не найдена. Проверьте корректность вызова инструмента."
                )
        else:
            final_answer_text = message_content.content

        answer_val = (
            extract_answer_from_reasoning(final_answer_text) if has_variants else None
        )
        print("answer_val:", answer_val)

        response_obj = PredictionResponse(
            id=request_id,
            answer=answer_val,
            reasoning=final_answer_text,
            sources=fix_broken_urls(validate_urls(sources_list)),
        )
        return response_obj

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка обработки запроса: {str(e)}")


@app.post("/api/request", response_model=PredictionResponse)
async def handle_request(body: PredictionRequest):
    """
    Эндпоинт для обработки POST запросов на /api/request.
    """
    return await predict(body)