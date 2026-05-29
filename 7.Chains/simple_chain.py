from langchain_openai import ChatOpenAI 
from dotenv import load_dotenv, parser
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv() # HuggingFace API key load

model = ChatOpenAI(model="gpt-4", temperature=0.9, max_completion_tokens=100) #temperature (0-2)[0: deterministic, 1: creative, 2: very creative]

parser = StrOutputParser()

template = PromptTemplate(
    template="Generate the name, age, and email of a fictional {place} person. \n",
    input_variables=["place"]
    )

chain = template | model | parser

print(chain.invoke({'place': "New York"}))

print(chain.get_graph().print_ascii())
