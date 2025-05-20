from langchain_ollama import ChatOllama
from langchain_ollama.embeddings import OllamaEmbeddings

llm = ChatOllama(
    model="deepseek-r1:1.5b",
    temperature=0,
)

embed = OllamaEmbeddings(
    model="nomic-embed-text",
)

input_texts = ["Document 1...", "Document 2..."]
vectors = embed.embed_documents(input_texts)
print("length:", len(vectors))
print("vectors:", vectors[0][:3])

messages = [
    (
        "system",
        "You are a helpful assistant that translates English to French. Translate the user sentence.",
    ),
    ("human", "I love programming."),
]
ai_msg = llm.invoke(messages)
print(ai_msg)
