from langchain_openai import ChatOpenAI 
from dotenv import load_dotenv, parser
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv() # HuggingFace API key load

model = ChatOpenAI(model="gpt-4", temperature=0.9, max_completion_tokens=100) #temperature (0-2)[0: deterministic, 1: creative, 2: very creative]

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template="Generate a proper report of {topic} \n",
    input_variables=["topic"]
    )

prompt2 = PromptTemplate(
    template="Generate a 5 pointer summary from the following text \n {text}",
    input_variables=["text"]
    )

chain = prompt1 | model | parser | prompt2 | model | parser

print(chain.invoke({'topic': "Global Warming"}))

print(chain.get_graph().print_ascii())