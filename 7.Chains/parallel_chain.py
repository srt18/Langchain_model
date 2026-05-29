from langchain_openai import ChatOpenAI 
from dotenv import load_dotenv, parser
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel

load_dotenv() # HuggingFace API key load

model1 = ChatOpenAI(model="gpt-4", temperature=0.9, max_completion_tokens=100) #temperature (0-2)[0: deterministic, 1: creative, 2: very creative]
model2 = ChatOpenAI(model="gpt-4", temperature=0.9, max_completion_tokens=100) #temperature (0-2)[0: deterministic, 1: creative, 2: very creative]

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template="Generate a short and simple note from the following text {text} \n",
    input_variables=["text"]
    )

prompt2 = PromptTemplate(
    template="Generate 5 short questions and answers from the following text \n {text}",
    input_variables=["text"]
    )

prompt3 = PromptTemplate(
    template="Merge the provided notes and quiz into a single document \n notes ->  {notes} and quiz -> {quiz}",
    input_variables=["notes", "quiz"]
    )

#Parallel chain - Runnable parallel
parrallel_chain = RunnableParallel({
    "notes": prompt1 | model1 | parser,
    "quiz": prompt2 | model2 | parser
 })

merge_chain = prompt3 | model1 | parser

chain = parrallel_chain | merge_chain

text = """Support vector machines (SVMs) are a set of supervised learning methods used for classification,
regression and outliers detection.

The advantages of support vector machines are:

. Effective in high dimensional spaces.
· Still effective in cases where number of dimensions is greater than the number of samples.
. Uses a subset of training points in the decision function (called support vectors), so it is also memory
efficient.

· Versatile: different Kernel functions can be specified for the decision function. Common kernels are
provided, but it is also possible to specify custom kernels.

The disadvantages of support vector machines include:

. If the number of features is much greater than the number of samples, avoid over-fitting in choosing
Kernel functions and regularization term is crucial.
. SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold
cross-validation (see Scores and probabilities, below).

The support vector machines in scikit-learn support both dense ( numpy.ndarray and convertible to that by
numpy.asarray) and sparse (any scipy. sparse) sample vectors as input. However, to use an SVM to make
predictions for sparse data, it must have been fit on such data. For optimal performance, use C-ordered
numpy.ndarray (dense) or scipy.sparse.csr_matrix (sparse) with dtype=float64"""

print(chain.invoke({'text': "Global Warming"}))

print(chain.get_graph().print_ascii())