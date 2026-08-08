from app.rag.pdf_rag import ask_report

question = input("Ask a question about the NVIDIA annual report: ")

results = ask_report(question)

print("\n===== RAG RESULTS =====\n")

for i, doc in enumerate(results, 1):
    print(f"\n--- Result {i} ---")
    print(doc.page_content[:1500])