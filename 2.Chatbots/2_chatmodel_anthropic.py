from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

model = ChatAnthropic(model="claude-3.5", temperature=0.9, max_completion_tokens=10) #temperature (0-2)[0: deterministic, 1: creative, 2: very creative]

result = model.invoke("What top 5 share in India?")

print(result.content)
