# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers import StrOutputParser, ResponseSchema
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_openai import ChatOpenAI 
from dotenv import load_dotenv, parser
from langchain_core.prompts import PromptTemplate

load_dotenv() # HuggingFace API key load

# llm = HuggingFaceEndpoint(
#     repo_id="google/gemma-2-2b-it", # Gemma-2b-it is a smaller version of Gemma-2b, optimized for inference tasks. It offers a good balance between performance and resource requirements, making it suitable for various applications.
#     task="text-generation",
#     huggingfacehub_api_token="HUGGINGFACEHUB_ACCESS_TOKEN"
# )

# model = ChatHuggingFace(llm=llm)

model = ChatOpenAI(model="gpt-4", temperature=0.9, max_completion_tokens=100) #temperature (0-2)[0: deterministic, 1: creative, 2: very creative]

schema = [
    ResponseSchema(name="fact_1", description="the fact 1 about the topic"),
    ResponseSchema(name="fact_2", description="the fact 2 about the topic"),
    ResponseSchema(name="fact_3", description="the fact 3 about the topic")
]

parser = StrOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template="Provide three interesting facts about {topic} in the following format: fact_1, fact_2, fact_3. \n{format_instructions}",
    input_variables=["topic"],
    partial_variables={"format_instructions": parser.get_format_instructions() }
)
prompt = template.invoke(topic="Python programming")

result = model.invoke(prompt)

final_result = parser.parse(result.content)

print(final_result)
