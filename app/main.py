from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from config.settings import GOOGLE_API_KEY

# 1️LLM
llm = GoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
        google_api_key=GOOGLE_API_KEY

)

# 2️ Prompt with history placeholder
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

# 3️LCEL chain
chain = prompt | llm | StrOutputParser()

# 4️ In-memory message history store
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# 5️ Wrap chain with memory
conversation = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

# 6️ Chat loop
print("Type 'exit' to stop\n")

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    response = conversation.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": "user_1"}}
    )

    print("AI:", response)







# import google.generativeai as genai

# genai.configure(api_key=GOOGLE_API_KEY)

# for m in genai.list_models():
#     print(m.name, m.supported_generation_methods)
