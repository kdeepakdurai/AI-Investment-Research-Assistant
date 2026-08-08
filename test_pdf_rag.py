from app.rag.pdf_rag import create_vector_store

pdf_path = input("Enter PDF path: ")

print("Loading annual report...")

create_vector_store(pdf_path)

print("\nRAG TEST COMPLETED SUCCESSFULLY!")