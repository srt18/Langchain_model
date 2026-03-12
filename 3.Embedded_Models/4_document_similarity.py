from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv() # OpenAI API key load

embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=300)

documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Critket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

query = 'tell me about Jasprit Bumrah'

# Generate embeddings for documents and query
document_embeddings = embeddings.embed_documents(documents)
query_embedding = embeddings.embed_query(query)

# Compute cosine similarity between query and documents
similarity_scores = cosine_similarity([query_embedding], document_embeddings)[0]
index, score = sorted(list(enumerate(similarity_scores)), key=lambda x: x[1])[-1]
print(query)
print(documents[index])
print("Similarity Scores:", score)
