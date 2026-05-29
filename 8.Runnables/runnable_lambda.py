from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableParallel, RunnableSequence, RunnablePassthrough, RunnableLambda

load_dotenv()
def word_count(joke: str) -> int:
    return len(joke.split())

model = ChatOpenAI(model='gpt-3.5-turbo', temperature=0.9)

prompt1 = PromptTemplate(
    template = 'Write a joke about {topic}?',
    input_variables = ['topic']
)

parser = StrOutputParser()

joke_gen_chain = RunnableSequence(prompt1, model, parser)

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'word_count': RunnableLambda(word_count)
})

final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

result = final_chain.invoke({'topic': 'programming'})
# print(result)

final_result = """{} \n word count - {} """.format(result['joke'], result['word_count'])

print(final_result)