from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableParallel, RunnableSequence

load_dotenv()

prompt1 = PromptTemplate(
    template = 'Generate a tweet about {topic}?',
    input_variables = ['topic']
)

prompt2 = PromptTemplate(
    template = 'Generate a LinkedIn post about {topic}?',
    input_variables = ['topic']
)

model = ChatOpenAI(model='gpt-3.5-turbo', temperature=0.9)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'tweet': RunnableSequence(prompt1, model, parser),
    'linkedin_post': RunnableSequence(prompt2, model, parser)
})

result = parallel_chain.invoke({'topic': 'artificial intelligence'})

print(result['tweet'])
print(result['linkedin_post'])