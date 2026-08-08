from langchain_community.tools import DuckDuckGoSearchRun


def research_company(company: str) -> str:
    """
    Research a company using web search.
    """

    search = DuckDuckGoSearchRun()

    query = f"""
    {company} company overview,
    headquarters,
    products and services,
    business model,
    major competitors,
    recent company developments
    """

    result = search.run(query)

    return f"""
COMPANY RESEARCH
================

Company: {company}

WEB RESEARCH
------------

{result}
"""