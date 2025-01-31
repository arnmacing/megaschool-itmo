import json
import os
import re
from datetime import datetime
from typing import Dict, List

from openai import AsyncOpenAI

from schemas.request import PredictionRequest, PredictionResponse
from utils.logger import setup_logger
from utils.utils import (call_openai_with_retry, clean_query, fix_broken_urls,
                         sanitize_user_input, validate_urls)

from .functions import search_links

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY не задан в переменных окружения.")

current_year = datetime.now().year
MODEL_NAME = "gpt-4"
BASE_URL = os.getenv("PROXY_URL", "")

client = AsyncOpenAI(api_key=OPENAI_API_KEY, base_url=BASE_URL if BASE_URL else None)


async def classify_relevance(query: str) -> Dict[str, str]:
    """
    Проверяет, относится ли запрос к ИТМО.
    Возвращает: {"is_itmo_relevant": bool, "reason": str}
    """
    classification_prompt = {
        "role": "system",
        "content": (
            "Ты — интеллектуальный классификатор запросов, связанных с университетом ИТМО. "
            "Определи, относится ли данный запрос к студенческой, учебной, преподавательской или научной жизни Университета ИТМО.\n\n"
            "✅ Примеры связанных запросов:\n"
            "- Вопросы о зачислении, экзаменах, стипендиях, учебных программах.\n"
            "- Вопросы про преподавателей, кафедры, расписание, кампусы.\n"
            "- Студенческая жизнь, научные исследования, конференции, гранты.\n\n"
            "❌ Примеры не связанных запросов:\n"
            "- Общие вопросы по математике, физике или программированию без явной привязки к ИТМО.\n\n"
            "Если запрос явно связан с ИТМО, верни JSON:\n"
            '{"is_itmo_relevant": true, "reason": "..."}\n'
            "Если не связан – верни JSON:\n"
            '{"is_itmo_relevant": false, "reason": "..."}'
        ),
    }
    messages = [classification_prompt, {"role": "user", "content": query}]
    response = await call_openai_with_retry(
        client.chat.completions.create, model=MODEL_NAME, messages=messages
    )
    result_text = response.choices[0].message.content
    try:
        return json.loads(result_text)
    except json.JSONDecodeError:
        return {"is_itmo_relevant": False, "reason": "Ошибка обработки запроса."}


system_message = {
    "role": "system",
    "content": (
        f"Ты — интеллектуальный агент, предоставляющий актуальную и подробную информацию об Университете ИТМО на {current_year} год.\n\n"
        "ИТМО — Федеральное государственное автономное образовательное учреждение высшего образования, также известное как "
        "Университет ИТМО, НИУ ИТМО, ИТМО или ITMO University.\n\n"
        "Твоя задача – предоставлять информацию на русском языке в формате JSON, строго по заданной структуре.\n\n"
        "# Структура JSON-ответа\n"
        "- **id**: числовое значение, соответствующее идентификатору запроса.\n"
        "- **answer**: числовое значение, содержащее правильный вариант ответа (если вопрос предполагает выбор вариантов). "
        "Если вариантов нет – значение должно быть null.\n"
        "- **reasoning**: подробное объяснение или дополнительная информация по запросу. В конце обязательно добавляй фразу: "
        "'Ответ сгенерирован моделью GPT-4.'\n"
        "- **sources**: список ссылок на официальые источники. Если источники не требуются – пустой список [].\n\n"
        "# Алгоритм работы\n"
        "1. **Анализ запроса**: Определи, содержит ли вопрос варианты (пронумерованные от 1 до 10). Если да, выбери правильный вариант.\n"
        "2. **Формирование reasoning**: Предоставь подробное обоснование, аргументируя выбор ответа, без использования извинений.\n"
        "3. **Поиск источников**: При необходимости приведи до трёх релевантных ссылок на официальные ресурсы ИТМО.\n"
        "4. **Формирование JSON-ответа**: Собери все данные в валидный JSON-объект согласно структуре.\n\n"
        "# Формат вывода\n"
        "Ответ должен быть представлен в формате JSON без дополнительного обрамления.\n\n"
        "# Примеры\n"
        "**Запрос:**\n"
        "```json\n"
        '{\n  "query": "В каком городе находится главный кампус Университета ИТМО?\\n1. Москва\\n2. Санкт-Петербург\\n3. Екатеринбург\\n4. Нижний Новгород",\n  "id": 1\n}\n'
        "```\n\n"
        "**Ответ:**\n"
        "```json\n"
        '{\n  "id": 1,\n  "answer": 2,\n  "reasoning": "Главный кампус Университета ИТМО находится в Санкт-Петербурге. Это подтверждается информацией с официального сайта. Ответ сгенерирован моделью GPT-4.",\n  "sources": [\n    "https://itmo.ru/ru/",\n    "https://abit.itmo.ru/"\n  ]\n}\n'
        "```\n\n"
        "Если запрос не относится к теме ИТМО, отвечай: 'Этот вопрос выходит за рамки моей специализации.' и возвращай JSON с answer: null и пустыми sources."
    ),
}

tools = [
    {
        "name": "classify_relevance",
        "description": "Определяет, относится ли запрос к ИТМО.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "name": "search_links",
        "description": "Поиск ссылок по теме ИТМО.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "max_results": {"type": "integer"},
            },
            "required": ["query"],
        },
    },
]


async def openai_chat_completion(messages, functions=None):
    async with AsyncOpenAI(
        api_key=OPENAI_API_KEY, base_url=BASE_URL if BASE_URL else None
    ) as client:
        req_data = {
            "model": MODEL_NAME,
            "messages": messages,
            "temperature": 0.3,
        }
        if functions:
            req_data["functions"] = functions
            req_data["function_call"] = "auto"
        response = await call_openai_with_retry(
            client.chat.completions.create, **req_data
        )
    return response


async def predict(body: PredictionRequest) -> PredictionResponse:
    logger = await setup_logger()
    request_id = body.id
    original_query = body.query
    await logger.info(f"Получен запрос: id={request_id}, query={original_query}")
    safe_query = sanitize_user_input(original_query)
    safe_query = clean_query(safe_query)
    await logger.info(f"Очищенный запрос: {safe_query}")
    has_variants = bool(re.search(r"\n\d+\.", safe_query))
    completion = await openai_chat_completion(
        messages=[system_message, {"role": "user", "content": safe_query}],
        functions=tools,
    )
    message_content = completion.choices[0].message
    final_answer_text = ""
    sources_list: List[str] = []
    while True:
        if getattr(message_content, "function_call", None):
            tool_call = message_content.function_call
            fn_name = tool_call.name
            fn_args = json.loads(tool_call.arguments or "{}")
            await logger.info(f"Функция вызвана: {fn_name} с аргументами: {fn_args}")
            if fn_name == "classify_relevance":
                classification_result = await classify_relevance(fn_args["query"])
                if not classification_result["is_itmo_relevant"]:
                    return PredictionResponse(
                        id=request_id,
                        answer=None,
                        reasoning="Этот вопрос выходит за рамки моей специализации. Ответ сгенерирован моделью GPT-4.",
                        sources=[],
                    )
                function_response_json = json.dumps(classification_result)
                second_completion = await openai_chat_completion(
                    [
                        system_message,
                        {"role": "user", "content": safe_query},
                        {
                            "role": "assistant",
                            "content": None,
                            "function_call": {
                                "name": "classify_relevance",
                                "arguments": json.dumps(fn_args),
                            },
                        },
                        {
                            "role": "function",
                            "name": "classify_relevance",
                            "content": function_response_json,
                        },
                    ],
                    functions=tools,
                )
                message_content = second_completion.choices[0].message
                continue
            elif fn_name == "search_links":
                await logger.info(
                    f"Вызван search_links для запроса: {fn_args.get('query', '')}"
                )
                search_result = await search_links(
                    query=fn_args.get("query", ""),
                    max_results=fn_args.get("max_results", 3),
                )
                search_result_json = json.dumps(search_result)
                third_completion = await openai_chat_completion(
                    [
                        system_message,
                        {"role": "user", "content": safe_query},
                        {
                            "role": "assistant",
                            "content": None,
                            "function_call": {
                                "name": "search_links",
                                "arguments": json.dumps(fn_args),
                            },
                        },
                        {
                            "role": "function",
                            "name": "search_links",
                            "content": search_result_json,
                        },
                    ],
                    functions=tools,
                )
                message_content = third_completion.choices[0].message
                if "links" in search_result:
                    sources_list = [link["link"] for link in search_result["links"][:3]]
                    await logger.info(f"Получены источники: {sources_list}")
                continue
        else:
            final_answer_text = message_content.content
            break
    await logger.info(f"Исходный ответ модели: {final_answer_text}")
    try:
        parsed_final = json.loads(final_answer_text)
        if all(key in parsed_final for key in ["id", "answer", "reasoning", "sources"]):
            parsed_final["sources"] = fix_broken_urls(
                validate_urls(parsed_final["sources"])
            )
            return PredictionResponse(
                id=parsed_final["id"],
                answer=parsed_final["answer"],
                reasoning=parsed_final["reasoning"],
                sources=parsed_final["sources"],
            )
    except json.JSONDecodeError:
        await logger.info(
            "Не удалось распарсить исходный ответ модели как JSON. Применяю дополнительную обработку."
        )
    return PredictionResponse(
        id=request_id,
        answer=None,
        reasoning="Ошибка формирования ответа. Ответ сгенерирован моделью GPT-4.",
        sources=[],
    )
