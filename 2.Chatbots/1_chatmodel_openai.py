from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv() # OpenAI API key load

model = ChatOpenAI(model="gpt-4", temperature=0.9, max_completion_tokens=10) #temperature (0-2)[0: deterministic, 1: creative, 2: very creative]

result = model.invoke("What top 5 share in India?")

print(result.content)
