from rag.product_search import ProductSearch
from shared.config import settings
from langchain_anthropic import ChatAnthropic
from langchain.prompts import PromptTemplate
import os
from pprint import pprint

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class FashionAssistant:
    def __init__(self):
        self.retriever = ProductSearch()
        self.llm_client = ChatAnthropic(model_name="claude-3-5-sonnet-20240620", api_key=settings.LLM_API_KEY, temperature=0.2, max_tokens=512)

    def sanitize_results(self, search_results):
        context = ""
        for doc in search_results:
            context = context + f"Productname: {doc['product_name']}\nbrand: {doc['brand']}\ncolor: {doc['color']}\ndescription: {doc['description']}\nprice in INR: {doc['price']}\n\n"
        return context
    

    def build_prompt_template(self):
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

    def rag(self, question):
        search_results = self.retriever.search(question, limit=5)
        context = self.sanitize_results(search_results)
        chain = self.build_prompt_template() | self.llm_client
        response = chain.invoke({"question": question, "context": context})
        if isinstance(response.content, tuple) and len(response.content) == 1:
            response = response.content[0]
        else:
            response = response.content
        return response


if __name__ == "__main__":
    
    llm = FashionAssistant()
    question = "I am a women and need business casual"
    print(f"Question: {question}")
    answer = llm.rag(question)
    if isinstance(answer, tuple) and len(answer) == 1:
        answer = answer[0]
    print("Answer:")
    print(answer)
    print()