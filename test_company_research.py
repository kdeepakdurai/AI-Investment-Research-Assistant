from app.agents.company_research_agent import research_company


company = input("Enter company name: ")

print("\nResearching company...\n")

result = research_company(company)

print(result)