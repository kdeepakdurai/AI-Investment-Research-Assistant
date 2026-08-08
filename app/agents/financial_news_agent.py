from langchain_community.tools import DuckDuckGoSearchRun


def search_financial_news(query: str) -> str:
    search = DuckDuckGoSearchRun()

    result = search.run(
        f"latest financial news about {query}"
    )

    return result