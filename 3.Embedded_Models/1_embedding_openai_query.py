from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv() # OpenAI API key load

OpenAIEmbeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=32)
query = "What is the capital of India?"
embedding = OpenAIEmbeddings.embed_query(query)
print(embedding)

