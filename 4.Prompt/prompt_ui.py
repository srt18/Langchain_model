from langchain_core.prompts import PromptTemplate, load_prompt
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import streamlit as st # type: ignore
import base64

def get_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

bg = get_base64("GenAi_image.png")

st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{bg}");
        background-size: cover;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

load_dotenv() # OpenAI API key load
model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
st.header('Research tool')

#User input for the prompt
# user_input = st.text_input('Enter your prompt:')

# if st.button('Submit'):
#     result = model.invoke(user_input)
#     st.write(result.content)

#Dropdown options for the prompt
paper_input = st.selectbox( "Select Research Paper Name", ["Select...", "Attention Is All You Need", 
    "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", 
    "Diffusion Models Beat GANs on Image Synthesis"] )
style_input = st.selectbox( "Select Explanation Style", ["Beginner-Friendly", "Technical", "CodeOriented",
                                                          "Mathematical"] ) 
length_input = st.selectbox( "Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)",
                                                            "Long (detailed explanation)"] )
template = load_prompt('template.json')

# #template
# template = PromptTemplate(
#     template="""Please summarize the research paper titled "{paper_input}" with the following 
#     specifications:
#     Explanation Style: {style_input}
#     Explanation Length: {length_input}
#     1. Mathematical Details: - Include relevant mathematical equations if present in the paper. - Explain the mathematical concepts using simple, intuitive code snippets 
#     where applicable. 
#     2. Analogies: - Use relatable analogies to simplify complex ideas. 
#     If certain information is not available in the paper, respond with: "Insufficient 
#     information available" instead of guessing. 
#     Ensure the summary is clear, accurate, and aligned with the provided style and 
#     length.""",
#     input_variables=["paper_input", "style_input", "length_input"],
#     validate_template=True
#     )
# fill the placeholders
# prompt = template.invoke(
#     {
#         'paper_input': paper_input,
#         'style_input': style_input,
#         'length_input': length_input
#     }
# )

if st.button('Summarize'):
    chain = template | model
    result = chain.invoke(
        {
            'paper_input': paper_input,
            'style_input': style_input,
            'length_input': length_input
        }
    )
    # result = model.invoke(prompt)
    st.write(result.content)
    # st.write("Welcome choose your dropdown options")