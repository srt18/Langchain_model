import random
class FakeLLM:
    def __init__(self, name):
        print('LLM initialized')


    def predict(self, prompt):
        
        response_list = [
            'delhi is the capital of india.',
            'paris is the capital of france.',
        ]
        return {'response':random.choice(response_list)}  # Return the first response

class FakePromptTemplate:
    def __init__(self, template, input_variables):
        print('PromptTemplate initialized')
        self.template = template
        self.input_variables = input_variables

    def format(self, input_dict):
        return self.template.format(**input_dict)
    

template = FakePromptTemplate(
    template="write a {length} poem about {country}?",
    input_variables=["length","country"]
)   

prompt = template.format({"country": "India", "length": 'short'})

llm = FakeLLM(name="FakeLLM")
response = llm.predict(prompt)
print(response)
