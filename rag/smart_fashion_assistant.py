from rag.product_search import ProductSearch
from shared.config import settings
from langchain_anthropic import ChatAnthropic
from langchain.prompts import PromptTemplate
import os
from pprint import pprint

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class SmartFashionAssistant:
    def __init__(self):
        self.retriever = ProductSearch()
        self.llm_client = ChatAnthropic(model_name="claude-3-5-sonnet-20240620", api_key=settings.ANTHROPIC_API_KEY, temperature=0.2, max_tokens=512)

    def sanitize_results(self, search_results):
        context = ""
        for doc in search_results:
            context = context + f"Productname: {doc['product_name']}\nbrand: {doc['brand']}\ncolor: {doc['color']}\ndescription: {doc['description']}\nprice in INR: {doc['price']}\n\n"
        return context
    
    def build_concept_prompt(self):
        concept_prompt = PromptTemplate(
            input_variables=["question"],
            template="""
                You are a smart e-commerce assistant.Extract 2 key product types, styles, colors or occasions that we can use to search a product catalog. Pls consider the Gender details
                Return them as a short comma-separated list. Respond with just the list and nothing else.
                Given this customer request:
                "{question}"
            """.strip()
        )
        return concept_prompt
    

    def build_recommendation_prompt(self):
        prompt_template = """
        You're a Fashion Advisor. Select the best suited product for the QUESTION based on the CONTEXT from the product catalog database.
        Use only the facts from the CONTEXT when answering the QUESTION. 
        Response should include product name, brand, gender, color, description and price.
        If the CONTEXT does not provide enough information, say "I don't know".
        Include a titbit about why the product is suitable.
        QUESTION: {question}

        CONTEXT: 
        {context}
        """.strip()
        return PromptTemplate(
            input_variables=["question", "context"],
            template=prompt_template,
        )
    

    def rag1(self, question):
        concept_chain = self.build_concept_prompt() | self.llm_client
        concepts = concept_chain.invoke({"question": question})
        print(f"Extracted concepts: {concepts}")
        search_results = self.retriever.search(concepts, limit=5)
        context = self.sanitize_results(search_results)
        recommendation_chain = self.build_recommendation_prompt() | self.llm_client
        response = recommendation_chain.invoke({"question": question, "context": context})
        if isinstance(response.content, tuple) and len(response.content) == 1:
            response = response.content[0]
        else:
            response = response.content
        return response
    
    def rag(self, question):
        concept_chain = self.build_concept_prompt() | self.llm_client
        concepts = concept_chain.invoke({"question": question})
        print(f"Extracted concepts: {concepts}")
        search_results = self.retriever.search(concepts, limit=5)
        context = self.sanitize_results(search_results)
        return context


if __name__ == "__main__":
    sfa = SmartFashionAssistant()
    question = "I am a women and need to dress for an indian wedding"
    print(f"Question: {question}")
    answer = sfa.rag(question)
    print("Answer:")
    print(answer)
    print()