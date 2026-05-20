# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI 
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv() # HuggingFace API key load

# llm = HuggingFaceEndpoint(
#     repo_id="google/gemma-2-2b-it", # Gemma-2b-it is a smaller version of Gemma-2b, optimized for inference tasks. It offers a good balance between performance and resource requirements, making it suitable for various applications.
#     task="text-generation",
#     huggingfacehub_api_token="HUGGINGFACEHUB_ACCESS_TOKEN"
# )

# model = ChatHuggingFace(llm=llm)

model = ChatOpenAI(model="gpt-4", temperature=0.9, max_completion_tokens=100) #temperature (0-2)[0: deterministic, 1: creative, 2: very creative]

# 1 st Prompt -> Detailed response  
template1 = PromptTemplate(
    template='Write a detailed report of {topic} in 100 words.',
    input_variables=['topic']
)

# 2nd Prompt -> Summary
template2 = PromptTemplate(
    template='Summarize the following text in 20 words: {text}',
    input_variables=['text']
)

prompt1 = template1.invoke({'topic': 'Black holes'})
                           
result = model.invoke(prompt1)

prompt2 = template2.invoke({'text': result.content})

result2 = model.invoke(prompt2)
print()

print("Detailed Report:", result2.content)

# StrOutput parser -> No schema, just string output

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({'topic': 'Black holes'})

print("StrOutputParser Result:", result)