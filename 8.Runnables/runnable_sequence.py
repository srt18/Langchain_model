from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence

load_dotenv()

prompt1 = PromptTemplate(
    template = 'Write a joke about {topic}?',
    input_variables = ['topic']
)

prompt2 = PromptTemplate(
    template = 'Explain the following joke in a concise way: {joke}',
    input_variables = ['joke']
)

model = ChatOpenAI(model='gpt-3.5-turbo', temperature=0.9)

parser = StrOutputParser()

chain = RunnableSequence(prompt1, model, parser, prompt2, model, parser)

print(chain.invoke({'topic': 'programming'}))