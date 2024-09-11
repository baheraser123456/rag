from langchain_community.llms import ollama
from langchain.prompts import FewShotPromptTemplate
from langchain.prompts import PromptTemplate
print('start genrate')
prblem_prompt = 'what are ingredients for making pizza?'
llm=ollama.Ollama(model='llama3')

examples = [
    {"recipe": "olives pizza", "ingredients": "\n".join([
                                    "* 1/2 pizza dough recipe",
                                    "* 1 tablespoon olive oil",
                                    "* 8 ounces cream cheese",
                                    "* 1/2 cup mozzarella cheese shredded",
                                    "* 1 tablespoon fresh parsley",
                                    "* 1/3 cup sliced green olives",
                                    "* 1/3 cup sliced black olives"
                                ])},
    {"recipe":"falafel", "ingredients": "\n".join(
                                [
                                    "* 1 cup dried chickpeas, soaked overnight",
                                    "* 1/2 cup onion",
                                    "* 1 cup parsley",
                                    "* 1 cup cilantro",
                                    "* 1 tablespoon fresh parsley",
                                    "* 1/3 cup sliced green olives",
                                    "* 1/3 cup sliced black olives",
                                    "* 3 garlic cloves",
                                    "* 1 tsp salt"
                                ]
                                )}
]
example_prompt = PromptTemplate(
    template="Recipe Name: {recipe}\nIngredients: {ingredients}",
   
)

prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    suffix="Recipe Name: {recipe}\n",

)
res= llm.invoke(prblem_prompt)
print(res)
print('------------------------')
prom_message=prompt.invoke({'recipe':'koshry'})

print(llm.invoke(prom_message))
#context prompt
template = """
you are a expert in programming 
Now explain the following concept in suitable words to person who don't know about programming in terms: 
{concept}
"""

prompt = PromptTemplate(template=template, input_variables=["concept"])
prompt_text = prompt.invoke(concept="llm")

response = llm(prompt_text)
print(response)
#devided prompt
template = """
Solve the following math problem step by step:

What is 48 divided by 6, plus 3?
"""

response = llm(template)
print(response) 
#zero_shot_prompt
prompt = "Translate the following sentence into arabic: 'I am learning how to use AI.'"
response = llm(prompt)

print(response)
#role baser
template = """
You are an experienced software engineer. Explain to a beginner how Python lists work.
"""

response = llm(template)
print(response)
#limited tokens
template = "Explain what machine learning is in 50 words or less."

response = llm(template)
print(response)
#metadata prompt
template = """
The year is 2024, and  Ai is a rapidly developing field. With this context, explain how Ai are different from 1990 to now.
"""

response = llm(template)
print(response)
#dynamic prompt
from langchain.prompts import PromptTemplate

template = """
Summarize the following article in three sentences:

{article}
"""

prompt = PromptTemplate(template=template, input_variables=["article"])

article_text = "Ai use llm to solve problems"
prompt_text = prompt.invoke({'article':article_text})

response = llm(prompt_text)
print(response)  # Output will be a summary of the article.
