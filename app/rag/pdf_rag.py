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


def create_vector_store(pdf_path):

    print("Loading PDF...")

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    print(f"Pages loaded: {len(documents)}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)

    print(f"Chunks created: {len(chunks)}")

    embeddings = FastEmbedWrapper()

    vector_store = Chroma(
        collection_name="nvidia_annual_report",
        embedding_function=embeddings,
        persist_directory="data/chroma"
    )

    batch_size = 25
    total = len(chunks)

    print("Starting document indexing...")

    for i in range(0, total, batch_size):

        batch = chunks[i:i + batch_size]

        vector_store.add_documents(batch)

        done = min(i + batch_size, total)

        print(f"Indexed {done}/{total} chunks")

    print("Documents added successfully!")

    return vector_store


def ask_report(question):

    embeddings = FastEmbedWrapper()

    vector_store = Chroma(
        collection_name="nvidia_annual_report",
        embedding_function=embeddings,
        persist_directory="data/chroma"
    )

    retriever = vector_store.as_retriever(
        search_kwargs={"k": 4}
    )

    return retriever.invoke(question)