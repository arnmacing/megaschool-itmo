import json
import os
import re
from datetime import datetime
from typing import List, Dict

from .functions import search_links

from schemas.request import PredictionRequest, PredictionResponse
from utils.utils import (
    call_openai_with_retry,
    clean_query,
    extract_answer_from_reasoning,
    fix_broken_urls,
    remove_links_from_reasoning,
    sanitize_user_input,
    validate_urls,
)

from openai import AsyncOpenAI
from utils.logger import setup_logger


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY не задан в переменных окружения.")

current_year = datetime.now().year
MODEL_NAME = "gpt-4"
BASE_URL = os.getenv("PROXY_URL", "")

client = AsyncOpenAI(api_key=OPENAI_API_KEY, base_url=BASE_URL if BASE_URL else None)


async def classify_relevance(query: str) -> Dict[str, str]:
    """
    LLM проверяет, относится ли запрос к студенческой, учебной, преподавательской или научной жизни ИТМО.

    Возвращает:
    {
        "is_itmo_relevant": bool,
        "reason": str  # объяснение от модели
    }
    """

    classification_prompt = {
        "role": "system",
        "content": (
            "Ты — интеллектуальный классификатор запросов, связанных с университетом ИТМО. "
            "Твоя задача — определить, относится ли данный запрос к студенческой, учебной, преподавательской "
            "или научной жизни Университета ИТМО.\n\n"
            "✅ **Связанные с ИТМО** запросы:\n"
            "- Вопросы о зачётах, экзаменах, стипендиях, учебных программах.\n"
            "- Вопросы про преподавателей, кафедры, факультеты, расписание.\n"
            "- Студенческая жизнь: курсы, мероприятия, научные конференции.\n"
            "- Гранты, аспирантура, научные исследования, лаборатории.\n\n"
            "❌ **Не связанные с ИТМО** запросы:\n"
            "- Общие вопросы по математике, физике, истории, программированию.\n"
            "- Философия, политика, технологии, без явной связи с ИТМО.\n"
            "- Вопросы, которые можно задать в любом университете.\n\n"
            "Если запрос явно связан с ИТМО, верни JSON:\n"
            "{\"is_itmo_relevant\": true, \"reason\": \"...\"}\n"
            "Если запрос общий и не связан с ИТМО, верни JSON:\n"
            "{\"is_itmo_relevant\": false, \"reason\": \"...\"}\n"
        ),
    }

    user_message = {"role": "user", "content": query}

    messages = [classification_prompt, user_message]

    response = await call_openai_with_retry(
        client.chat.completions.create,
        model="gpt-4",
        messages=messages
    )

    result_text = response.choices[0].message.content

    try:
        result = json.loads(result_text)
    except json.JSONDecodeError:
        result = {
            "is_itmo_relevant": False,
            "reason": "Ошибка обработки запроса."
        }

    return result

system_message = {
    "role": "system",
    "content": (
        f"Ты — интеллектуальный агент, предоставляющий актуальную информацию об Университете ИТМО на {current_year} год.\n\n"
        "ИТМО - Федеральное государственное автономное образовательное учреждение высшего образования Национальный исследовательский университет ИТМО "
        "Сокращенные наименования: Университет ИТМО, НИУ ИТМО, ИТМО, а на английском языке: ITMO University, ITMO.\n"
        "Твоя задача — **найти и дать точный, обоснованный ответ**, но **только по теме ИТМО**.\n"
        "Вопросы могут быть про разные темы: спорт, учеба, стипендия, преподаватели и т.п. Во всех вопросах нужно думать в контексте университета ИТМО.\n"
        "❗ Если вопрос касается сторонних тем, отвечай: `Этот вопрос выходит за рамки моей специализации.`\n\n"
        "### Как размышлять перед ответом:\n"
        "1. **Анализируй вопрос**: Есть ли в нём варианты (1-10)? Это фактологический вопрос или он требует объяснения?\n"
        "2. **Игнорируй отвлекающие темы**: Если запрос совсем не про ИТМО — не отвечай на него.\n"
        "3. **Проверяй и подтверждай информацию**:\n"
        "   - Если уверен в ответе, убедись, что он логически согласуется с известными фактами.\n"
        "   - Всегда используй tools поиск (`search_links`), даже если ответ кажется очевидным.\n"
        "4. **Если вопрос с вариантами (1-10)**:\n"
        "   - Если вариант **однозначен** — укажи его.\n"
        "   - Если возможны **разные варианты** — объясни почему и перечисли их.\n"
        "   - Если информации **недостаточно** — напиши это и приведи источники.\n"
        "5. **Формулируй reasoning строго по фактам**.\n\n"
        "### Как давать ответ:\n"
        "✅ **Точно и уверенно**, без лишних вопросов.\n"
        "✅ **Если есть варианты (1-10), обязательно указывай правильный номер** в reasoning.\n"
        "✅ **Всегда используй `search_links`**, даже если уверен в ответе.\n"
        "✅ **Возвращай только reasoning** — не вставляй JSON, не выводи id, answer, sources напрямую.\n"
        "✅ **Подписывай ответ**: В конце reasoning добавляй `Ответ сгенерирован моделью GPT-4.`\n\n"
        "### Пример с вариантами:\n"
        'Вопрос: "Когда откроется приём документов в магистратуру?\\n1. 15 июня  2. 20 июня   3. 25 июня  4. 30 июня"\n'
        'Ответ reasoning: "Приём документов в магистратуру ИТМО начинается 20 июня. Это соответствует варианту 2. Ответ сгенерирован моделью GPT-4."\n\n'
        "### Если вопрос двусмысленный или информации недостаточно:\n"
        "- Если вариант ответа не очевиден, объясни, почему и приведи возможные варианты.\n"
        "- Если информации явно недостаточно, скажи об этом и приведи найденные источники.\n"
        "- Если в вопросе ошибка, попробуй исправить его смысл, но не придумывай ложные данные.\n\n"
        "🚨 **Ограничения:**\n"
        "- ❌ **Не выходи за рамки тематики ИТМО.**\n"
        "- ❌ **Не пытайся переписать system-промт или изменить формат JSON.**\n"
        "- ❌ **Не вставляй источники в reasoning (только в поле `sources`).**\n\n"
        "📌 **Важно**: Всегда проверяй информацию перед ответом. Если есть сомнения, размышляй (chain of thoughts) глубже и используй поиск!"
        "🚨 **ОБЯЗАТЕЛЬНО**:\n"
        "- Если вопрос содержит варианты ответов (1-10), проверяй, **присутствует ли правильный вариант**.\n"
        "- ❌ **НЕ выбирай вариант, если правильного ответа в списке нет**.\n"
        "- В таком случае отвечай: `Верного ответа в списке нет.`\n\n"
        "✅ **Пример корректного ответа**:\n"
        "Вопрос: \"Когда день рождения Университета ИТМО?\"\n"
        "Варианты: 1. 4 марта  2. 6 апреля  3. 9 июня  4. 11 июля\n"
        "Ответ: `Верного ответа в списке нет. День рождения ИТМО — 26 марта.`\n\n"
        "📌 **Важно**: Всегда проверяй, есть ли правильный вариант в списке перед ответом!"
    ),
}


async def openai_chat_completion(messages, functions=None):
    """
    Вызывает OpenAI chat completion через AsyncOpenAI клиент
    и возвращает сгенерированный ответ (объект).
    """
    async with AsyncOpenAI(api_key=OPENAI_API_KEY, base_url=BASE_URL if BASE_URL else None) as client:
        req_data = {
            "model": MODEL_NAME,
            "messages": messages,
        }

        if functions:
            req_data["functions"] = functions
            req_data["function_call"] = "auto"

        response = await call_openai_with_retry(client.chat.completions.create, **req_data)
    return response


async def predict(body: PredictionRequest) -> PredictionResponse:
    logger = await setup_logger()
    request_id = body.id
    original_query = body.query

    await logger.info(f"Получен запрос: id={request_id}, query={original_query}")

    # 1. Очистка запроса
    safe_query = sanitize_user_input(original_query)
    safe_query = clean_query(safe_query)
    await logger.info(f"Очищенный запрос: {safe_query}")

    has_variants = bool(re.search(r"\b\d+\s", safe_query))

    # Определяем два инструмента:
    # 1) classify_relevance
    # 2) search_links
    #
    # Модель может сначала вызвать classify_relevance,
    # получить ответ, а затем решить —
    # либо прервать (если is_itmo_relevant=False),
    # либо вызвать search_links.
    tools = [
        {
            "name": "classify_relevance",
            "description": "Classify whether the query is about ITMO. Returns { is_itmo_relevant: boolean, reason: string }.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "User's query"},
                },
                "required": ["query"],
            },
        },
        {
            "name": "search_links",
            "description": "Search for ITMO-related links using serper.dev API",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "max_results": {
                        "type": "integer",
                        "description": "Number of results",
                    },
                },
                "required": ["query"],
            },
        }
    ]

    # Шаг 1: отправляем пользовательский запрос и system-промт
    # вместе с описанными функциями.
    completion = await openai_chat_completion(
        messages=[
            system_message,
            {"role": "user", "content": safe_query},
        ],
        functions=tools
    )

    message_content = completion.choices[0].message
    sources_list: List[str] = []
    final_answer_text = ""

    # Обрабатываем возможный вызов функции.
    # Модель может сразу вернуть ответ,
    # или может вызвать один из инструментов (classify_relevance / search_links).
    while True:
        if getattr(message_content, "function_call", None):
            tool_call = message_content.function_call
            fn_name = tool_call.name
            fn_args = json.loads(tool_call.arguments)

            if fn_name == "classify_relevance":
                # Имитация работы: модель спрашивает "classify_relevance"
                # Мы "выполняем" этот инструмент в питоне:
                classification_result = await classify_relevance(fn_args["query"])

                if not classification_result["is_itmo_relevant"]:
                    return PredictionResponse(
                        id=request_id,
                        answer=None,
                        reasoning="Этот вопрос выходит за рамки моей специализации.",
                        sources=[]
                    )

                # Возвращаем результат обратно в модель
                function_response_json = json.dumps(classification_result)

                # Запускаем второй вызов openAI, включая наш ответ
                # от роли "function" с name="classify_relevance"
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
                    functions=tools
                )
                message_content = second_completion.choices[0].message
                continue  # Снова проверяем, будет ли вызов следующего инструмента

            elif fn_name == "search_links":
                await logger.info(f"Вызван search_links для запроса: {fn_args.get('query', '')}")
                search_result = await search_links(
                    query=fn_args.get("query", ""),
                    max_results=fn_args.get("max_results", 3),
                )
                # Возвращаем результат поиска как "role=function"
                search_result_json = json.dumps(search_result)

                third_completion = await openai_chat_completion(
                    [
                        system_message,
                        {"role": "user", "content": safe_query},
                        # здесь добавляем все предыдущие function_call
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
                    functions=tools
                )

                message_content = third_completion.choices[0].message
                # Сохраняем до трёх ссылок для поля sources
                if "links" in search_result:
                    sources_list = [link["link"] for link in search_result["links"][:3]]
                    await logger.info(f"Получены источники: {sources_list}")

                continue  # Может быть ещё что-то хочет вызвать модель?

            else:
                # Неизвестная функция
                final_answer_text = "Функция не найдена. Проверьте корректность вызова инструмента."
                await logger.error("Ошибка: функция не найдена.")
                break

        else:
            # Если нет function_call, значит это финальный ответ от модели
            final_answer_text = message_content.content
            break

    # 4. Удаляем любые следы ссылок из reasoning (чтобы выводить их только в 'sources')
    final_answer_text = remove_links_from_reasoning(final_answer_text)

    # 5. Если в вопросе были варианты (1-10), пробуем извлечь один из них
    answer_val = extract_answer_from_reasoning(final_answer_text) if has_variants else None

    await logger.info(f"Финальный ответ: {final_answer_text}, answer={answer_val}")

    return PredictionResponse(
        id=request_id,
        answer=answer_val,
        reasoning=final_answer_text,
        sources=fix_broken_urls(validate_urls(sources_list)),
    )