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