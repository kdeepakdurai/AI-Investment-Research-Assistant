from app.agents.financial_news_agent import search_financial_news


query = input("Enter company or financial topic: ")

print("\nSearching financial news...\n")

result = search_financial_news(query)

print(result)