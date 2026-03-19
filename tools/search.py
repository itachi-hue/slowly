from __future__ import annotations

import asyncio
import re
import time
from typing import Optional

import requests
from bs4 import BeautifulSoup


def _clean_text(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    return s


async def web_search(query: str, max_results: int = 5, tavily_api_key: Optional[str] = None) -> list[dict]:
    """
    Priority:
    1) Tavily (if key provided)
    2) DuckDuckGo via duckduckgo-search (imported lazily)
    """
    if tavily_api_key:
        try:
            from tavily import TavilyClient  # type: ignore

            def _tavily() -> list[dict]:
                client = TavilyClient(api_key=tavily_api_key)
                resp = client.search(query=query, max_results=max_results)
                out: list[dict] = []
                for r in resp.get("results", [])[:max_results]:
                    out.append({"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("content", "")})
                return out

            return await asyncio.to_thread(_tavily)
        except Exception:
            # fall through to DDG
            pass

    from duckduckgo_search import DDGS  # type: ignore

    def _ddg() -> list[dict]:
        out: list[dict] = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                out.append({"title": r.get("title", ""), "url": r.get("href", ""), "snippet": r.get("body", "")})
        return out[:max_results]

    return await asyncio.to_thread(_ddg)


async def fetch_page(url: str, timeout_s: int = 20) -> str:
    def _fetch() -> str:
        headers = {"User-Agent": "overnight-agent/0.1 (+local)"}
        r = requests.get(url, headers=headers, timeout=timeout_s)
        r.raise_for_status()
        html = r.text
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = soup.get_text(" ")
        return _clean_text(text)

    return await asyncio.to_thread(_fetch)


async def search_and_fetch(
    query: str,
    max_results: int = 5,
    tavily_api_key: Optional[str] = None,
    per_page_char_limit: int = 12000,
    max_concurrency: int = 5,
) -> dict:
    t0 = time.time()
    results = await web_search(query=query, max_results=max_results, tavily_api_key=tavily_api_key)
    sem = asyncio.Semaphore(max_concurrency)

    async def _one(res: dict) -> dict:
        async with sem:
            try:
                url = str(res.get("url", ""))
                txt = await fetch_page(url)
                return {
                    "title": res.get("title", ""),
                    "url": url,
                    "snippet": res.get("snippet", ""),
                    "content": txt[:per_page_char_limit],
                }
            except Exception as e:
                return {
                    "title": res.get("title", ""),
                    "url": res.get("url", ""),
                    "snippet": res.get("snippet", ""),
                    "error": str(e),
                }

    pages = await asyncio.gather(*[_one(r) for r in results])
    return {"query": query, "elapsed_s": round(time.time() - t0, 3), "results": pages}


