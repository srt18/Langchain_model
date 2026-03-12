from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation" ,
    pipeline_kwargs= dict(
        max_new_tokens=2048, 
        temperature= 0.7)
    )

model = ChatHuggingFace(llm=llm)
result = model.invoke("Name 5 cities of India?")
print(result.content)
