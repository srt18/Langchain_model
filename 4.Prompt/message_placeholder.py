from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


#chat template
chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="chat_history"),
    ('human', "{query}"),
])

chat_history = []
# load chat history
with open('4.Prompt/chat_history.txt', 'r') as f:
    chat_history.extend(f.readlines())

#create prompt
prompt = chat_template.invoke({'chat_history': chat_history,
                      'query': 'Where is my refund?'})
print("Prompt:", prompt)