
from rag.product_search import ProductSearch
from shared.config import settings
from langchain_anthropic import ChatAnthropic
from langchain.prompts import PromptTemplate
import os
from pprint import pprint

os.environ["TOKENIZERS_PARALLELISM"] = "false"

llm_client = ChatAnthropic(model_name="claude-3-5-sonnet-20240620", api_key=settings.ANTHROPIC_API_KEY, temperature=0.2, max_tokens=512)

prompt ="""
                You are a stylist + product search planner. 
                Classify the user's question and understand what they are looking for and extract 2 key product types, styles, colors or occasions that we can use to search a product catalog. 
                Pls consider the Gender details
                Return them as a short comma-separated list. Respond with just the list and nothing else.
                Given this customer questions:
                "{question}"
                Response should be comma-separated list like "options1, options"
                If the question is irrelevant to the Fashion products catalog, respond with " "
        """

def build_concept_prompt():
    concept_prompt = PromptTemplate(
        input_variables=["question"],
        template=prompt.strip()
    )
    return concept_prompt
    

def rag(question):
    concept_chain = build_concept_prompt() | llm_client
    response = concept_chain.invoke({"question": question})
    return response


if __name__ == "__main__":
    q = input("Enter your fashion question: ")
    #q="I am a women and need to dress for an indian weddings"
    response = rag(q)
    print(f"Extracted concepts: {response.content}")
    if len(response.content) > 0:
        l = response.content.split(",")
        print(f"Concepts extracted successfully.{l}")
    else:
        print("No concepts found.")