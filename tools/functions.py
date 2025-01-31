import os
from typing import Any, Dict

import httpx
from dotenv import load_dotenv

load_dotenv()

SERPER_API_KEY = os.getenv("SERPER_API_KEY")
SERPER_API_URL = "https://google.serper.dev/search"


def expand_query(query: str) -> str:
    """
    Если в запросе нет слова 'итмо', добавляем 'Университет ИТМО'
    для повышения релевантности поиска.
    """
    if "итмо" not in query.lower():
        query += " Университет ИТМО"
    return query


async def search_links(query: str, max_results: int = 5) -> Dict[str, Any]:
    """
    Выполняет поиск через Serper.dev API (Google Search) и возвращает ссылки с
    заголовками и краткими описаниями. Сортирует результаты, отдавая приоритет
    официальным (itmo.ru, news.itmo.ru) и тем, у которых упоминается 'итмо'
    в заголовке или сниппете.
    """
    if not SERPER_API_KEY:
        return {"error": "SERPER_API_KEY не задан."}

    query = expand_query(query)

    headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
    params = {
        "q": query,
        "num": max_results * 3,
        "gl": "ru",
        "hl": "ru",
        "googleDomain": "google.ru",
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(SERPER_API_URL, headers=headers, json=params)
        response.raise_for_status()
        data = response.json()

        if "organic" not in data:
            return {"links": []}

        results = []
        for item in data["organic"]:
            title = item.get("title", "")
            link = item.get("link", "")
            snippet = item.get("snippet", "")

            priority = 0
            if "itmo.ru" in link.lower():
                priority += 2
                if "news.itmo.ru" in link.lower():
                    priority += 1

            if "итмо" in title.lower() or "итмо" in snippet.lower():
                priority += 1

            results.append(
                {
                    "title": title,
                    "link": link,
                    "snippet": snippet,
                    "priority": priority,
                }
            )

        sorted_results = sorted(
            results, key=lambda x: (x["priority"], -len(x["snippet"])), reverse=True
        )

        return {"links": sorted_results[:max_results]}

    except httpx.HTTPStatusError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": str(e)}
