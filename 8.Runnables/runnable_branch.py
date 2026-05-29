from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence, RunnablePassthrough, RunnableBranch

load_dotenv()

prompt1 = PromptTemplate(
    template = 'Write a detailed report about {topic}?',
    input_variables = ['topic']
)

prompt2 = PromptTemplate(
    template = 'Summarize the following text about {text}?',
    input_variables = ['text']
)

model = ChatOpenAI(model='gpt-3.5-turbo', temperature=0.9)
parser = StrOutputParser()

report_chain = RunnableSequence(prompt1, model, parser)

branch_chain = RunnableBranch(
    (lambda x : len(x.split()) > 200, RunnableSequence(prompt2, model, parser)),
    RunnablePassthrough() 
)

final_chain = RunnableSequence(report_chain, branch_chain)

print(final_chain.invoke({'topic': 'russia vs ukraine war'}))