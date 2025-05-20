from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="deepseek-r1:1.5b",
    temperature=0,
)

messages = [
    (
        "system",
        "You are a helpful assistant that translates English to French. Translate the user sentence.",
    ),
    ("human", "I love programming."),
]
ai_msg = llm.invoke(messages)
print(ai_msg)
