import os
from dotenv import load_dotenv

import weaviate
from weaviate.embedded import EmbeddedOptions

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings



def load_env() -> None:
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Falta OPENAI_API_KEY en .env")


def create_llm() -> ChatOpenAI:
    # El libro usa gpt-3.5-turbo y temperature=0【turn1file3†L71-L71】
    return ChatOpenAI(model="gpt-3.5-turbo", temperature=0)


def create_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings()


def create_weaviate_client() -> weaviate.Client:
    # Weaviate embebido (sin servidor externo)【turn1file3†L58-L61】
    return weaviate.Client(embedded_options=EmbeddedOptions())
