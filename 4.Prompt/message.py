from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv() # OpenAI API key load

model = ChatOpenAI()

message = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="tell me about langchain")
]

result = model.invoke(message)

message.append(AIMessage(content=result.content))

print("Message:", message)