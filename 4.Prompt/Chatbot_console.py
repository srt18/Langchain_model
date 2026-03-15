from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage 
from dotenv import load_dotenv

load_dotenv() # OpenAI API key load

model = ChatOpenAI()
chat_history = [
    SystemMessage(content="You are a helpful AI assistant.")
]

while True:
    user_input = input("You: ")
    # chat_history.append({"role": "user", "content": user_input})
    chat_history.append(HumanMessage(content=user_input))
    if user_input == "exit":
        print("Exiting the chatbot. Goodbye!")
        break
    result = model.invoke(user_input)
    # chat_history.append({"role": "assistant", "content": result.content})
    chat_history.append(AIMessage(content=result.content))
    print("Chatbot:", result.content)

print("Chat History:", chat_history)