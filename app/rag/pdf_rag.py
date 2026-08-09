from pathlib import Path
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings
from fastembed import TextEmbedding


class FastEmbedWrapper(Embeddings):

    def __init__(self):
        self.model = TextEmbedding(
            model_name="BAAI/bge-small-en-v1.5"
        )

    def embed_documents(self, texts):
        return list(self.model.embed(texts))

    def embed_query(self, text):
        return list(self.model.embed([text]))[0]


BASE_DIR = Path(__file__).resolve().parents[2]

PDF_PATH = BASE_DIR / "data" / "uploads" / "annual_report.pdf"
CHROMA_PATH = BASE_DIR / "data" / "chroma"


@lru_cache(maxsize=1)
def get_vector_store():

    embeddings = FastEmbedWrapper()

    vector_store = Chroma(
        collection_name="nvidia_annual_report",
        embedding_function=embeddings,
        persist_directory=str(CHROMA_PATH)
    )

    count = vector_store._collection.count()

    print(f"Chroma documents: {count}")

    if count == 0:

        print("Chroma database is empty.")
        print("Building RAG database from annual report...")

        if not PDF_PATH.exists():
            raise FileNotFoundError(
                f"Annual report not found: {PDF_PATH}"
            )

        loader = PyPDFLoader(str(PDF_PATH))
        documents = loader.load()

        print(f"Pages loaded: {len(documents)}")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )

        chunks = splitter.split_documents(documents)

        print(f"Chunks created: {len(chunks)}")

        batch_size = 25
        total = len(chunks)

        for i in range(0, total, batch_size):

            batch = chunks[i:i + batch_size]

            vector_store.add_documents(batch)

            done = min(i + batch_size, total)

            print(f"Indexed {done}/{total} chunks")

        print("RAG database created successfully!")

    return vector_store


def create_vector_store(pdf_path):

    embeddings = FastEmbedWrapper()

    vector_store = Chroma(
        collection_name="nvidia_annual_report",
        embedding_function=embeddings,
        persist_directory=str(CHROMA_PATH)
    )

    if vector_store._collection.count() > 0:
        return vector_store

    loader = PyPDFLoader(str(pdf_path))
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)

    batch_size = 25

    for i in range(0, len(chunks), batch_size):

        batch = chunks[i:i + batch_size]

        vector_store.add_documents(batch)

        print(
            f"Indexed {min(i + batch_size, len(chunks))}/{len(chunks)} chunks"
        )

    return vector_store


def ask_report(question):

    vector_store = get_vector_store()

    retriever = vector_store.as_retriever(
        search_kwargs={"k": 4}
    )

    return retriever.invoke(question)