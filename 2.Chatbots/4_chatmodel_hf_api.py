from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv() # HuggingFace API key load

llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    huggingfacehub_api_token="HUGGINGFACEHUB_ACCESS_TOKEN"
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("Name 5 cities of India?")

print(result.content)
