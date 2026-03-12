from langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv() # OpenAI API key load

# Initialize the OpenAI LLM with the specified model
llm = OpenAI(model="gpt-3.5-turbo-instruct") 
# Invoke the LLM with a prompt to get the answer to the question about the capital of India
result = llm.invoke("What is the capital of India?") 

print(result)
