from langchain_openai import ChatOpenAI 
from dotenv import load_dotenv, parser
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv() # HuggingFace API key load

model1 = ChatOpenAI(model="gpt-4", temperature=0.9, max_completion_tokens=100) #temperature (0-2)[0: deterministic, 1: creative, 2: very creative]
# model2 = ChatOpenAI(model="gpt-4", temperature=0.9, max_completion_tokens=100) #temperature (0-2)[0: deterministic, 1: creative, 2: very creative]

parser1 = StrOutputParser()

class feedback(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(description="The sentiment of the feedback")

parser2 = PydanticOutputParser(pydantic_object=feedback)

prompt1 = PromptTemplate(
    template="Classify the sentiment of the following {feedback} text into positive or negative \n {format_instructions}",
    input_variables=["feedback"],
    partial_variables={'format_instructions': parser2.get_format_instructions()}
    )

classifier_chain = prompt1 | model1 | parser2

prompt2 = PromptTemplate(   
    template="Write an appropriate response to this positive feedback: {feedback} \n",
    input_variables=["feedback"]
    )

prompt3 = PromptTemplate(   
    template="Write an appropriate response to this negative feedback: {feedback} \n",
    input_variables=["feedback"]
    )

#Runnable Branch -- IF , elif , else concept
branch_chain = RunnableBranch(
    (lambda x: x.sentiment == "positive", prompt2 | model1 | parser1),
    (lambda x: x.sentiment == "negative", prompt3 | model1 | parser1),
    RunnableLambda(lambda x: "Could not find the sentiment of the feedback")
)

chain = classifier_chain | branch_chain

# print(classifier_chain.invoke({"feedback": "The product is really bad and I am not satisfied with it."}).sentiment)  

print(chain.invoke({"feedback": "The product is terrible product."}))
print(chain.get_graph().print_ascii())
