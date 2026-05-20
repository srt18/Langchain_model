# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import JsonOutputParser
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
parser = JsonOutputParser()

template = PromptTemplate(
    template="What is the capital of France? \n {format_instructions}",
    input_variables=[],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# prompt = template.format_prompt()

# print(prompt)

# result = model.invoke(prompt)
# final_result = parser.parse(result.content)

chain = template | model | parser
final_result = chain.invoke({})
print(final_result)
print(type(final_result))