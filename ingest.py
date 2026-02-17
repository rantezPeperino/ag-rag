import requests
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

from langchain_community.vectorstores import Weaviate


def download_state_of_union(output_path: str = "state_of_the_union.txt") -> str:
    # El libro baja el archivo desde GitHub y lo guarda local【turn1file3†L46-L52】
    url = "https://frontiernerds.com/files/state_of_the_union.txt"
    res = requests.get(url, timeout=30)
    res.raise_for_status()

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(res.text)

    return output_path


def build_retriever(
    weaviate_client,
    embeddings,
    source_path: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
):
    loader = TextLoader(source_path, encoding="utf-8")
    documents = loader.load()

    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(documents)

    # El libro usa Weaviate.from_documents(..., by_text=False)【turn1file3†L62-L67】
    vectorstore = Weaviate.from_documents(
        client=weaviate_client,
        documents=chunks,
        embedding=embeddings,
        by_text=False,
    )

    return vectorstore.as_retriever()
