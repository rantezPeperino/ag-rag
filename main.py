from config import load_env, create_llm, create_embeddings, create_weaviate_client
from input_data import create_rag_prompt
from ingest import download_state_of_union, build_retriever
from pipeline import build_app, run_query


def main():
    load_env()

    llm = create_llm()
    embeddings = create_embeddings()
    client = create_weaviate_client()
    prompt = create_rag_prompt()

    path = download_state_of_union("state_of_the_union.txt")
    retriever = build_retriever(client, embeddings, path)

    app = build_app(llm, retriever, prompt)

    q1 = "What did the president say about Justice Breyer"
    out1 = run_query(app, q1)
    print("\nQ1:", q1)
    print(out1)

    q2 = "What did the president say about the economy?"
    out2 = run_query(app, q2)
    print("\nQ2:", q2)
    print(out2)


if __name__ == "__main__":
    main()
