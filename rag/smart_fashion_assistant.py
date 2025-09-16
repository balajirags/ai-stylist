from rag.product_search import ProductSearch
from shared.config import settings
from langchain_anthropic import ChatAnthropic
from langchain.prompts import PromptTemplate
import os
import json

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class SmartFashionAssistant:
    def __init__(self):
        self.retriever = ProductSearch()
        self.llm_client = ChatAnthropic(model_name="claude-3-5-sonnet-20240620", api_key=settings.LLM_API_KEY, temperature=0.2, max_tokens=512)

    def sanitize_results(self, search_results):
        context = ""
        for doc in search_results:
            context = context + f"Productname: {doc['product_name']}\nbrand: {doc['brand']}\ncolor: {doc['color']}\ndescription: {doc['description']}\nprice in INR: {doc['price']}\n\n"
        return context
    
    def build_concept_prompt(self):
        concept_prompt = PromptTemplate(
            input_variables=["question"],
            template="""
                You are a stylist + product search planner. 
                Classify the user's question and understand what they are looking for and extract 2 key product types, styles, colors or occasions that we can use to search a product catalog. 
                Pls consider the Gender details
                Return them as a short comma-separated list. Respond with just the list and nothing else.
                Given this customer questions:
                "{question}"
                Response should be comma-separated list like "options1, options2"
                If the question is irrelevant to the Fashion products catalog, respond with " "
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
    
    def build_evaluation_prompt(self):
        evaluation_prompt = PromptTemplate(
            input_variables=["question", "answer", "context"],
            template="""
                You are an expert evaluator for a RAG system.
                Your task is to analyze the relevance of the generated answer to the given question.
                Based on the relevance of the generated answer, you will classify it
                as "NON_RELEVANT", "PARTLY_RELEVANT", or "RELEVANT".

                Here is the data for evaluation:

                Question: {question}
                Generated Answer: {answer}

                Please analyze the content and context of the generated answer in relation to the question
                and provide your evaluation in parsable JSON without using code blocks:
                {{
                    "Relevance": "NON_RELEVANT" | "PARTLY_RELEVANT" | "RELEVANT",
                    "Explanation": "[Provide a brief explanation for your evaluation]"
                }}
            """.strip()
        )
        return evaluation_prompt
    

    def evaluate_relevance(self, question, answer):
        evaluation_chain = self.build_evaluation_prompt() | self.llm_client
        evaluation_response = evaluation_chain.invoke({"question": question, "answer": answer})
        content, metadata = evaluation_response.content, evaluation_response.usage_metadata
        try:
            json_eval = json.loads(content)
            return json_eval,metadata, evaluation_response
        except json.JSONDecodeError:
            result = ({"Relevance": "UNKNOWN", "Explanation": "Failed to parse evaluation"}, {"total_tokens": 0, "input_tokens": 0, "output_tokens": 0}, evaluation_response)
            return result

    def extract_concept(self, content):
        if isinstance(content, str) and len(content.strip()) > 0:
            return [item.strip() for item in content.split(",") if item.strip()]
        return []

    def rag(self, question):
        concept_chain = self.build_concept_prompt() | self.llm_client
        concepts_response = concept_chain.invoke({"question": question})
        print(f"Extracted concepts: {concepts_response.content}")
        product_types = self.extract_concept(concepts_response.content)
        if len(product_types) > 0:
            search_results = self.retriever.multi_query_hybrid_search(product_types, limit=5)
            context = self.sanitize_results(search_results)
            recommendation_chain = self.build_recommendation_prompt() | self.llm_client
            response = recommendation_chain.invoke({"question": question, "context": context})
            response_content = response.content
            metadata = response.usage_metadata
            if isinstance(response.content, tuple) and len(response.content) == 1:
                response_content = response.content[0]
            else:
                response_content = response.content
            return (response_content, metadata, response)
        return ("Question seems irrelevant to the Fashion products catalog.", {"total_tokens": 0, "input_tokens": 0, "output_tokens": 0})

"""
if __name__ == "__main__":
    sfa = SmartFashionAssistant()
    question = "I am a women and need to dress for an indian wedding?"
    print(f"Question: {question}")
    content, metadata, response = sfa.rag(question)
    print("Answer:")
    print(content)
    print()
    print("metadata:")
    print(f"Input token - {metadata['input_tokens']}, Output tokens - {metadata['output_tokens']}, Total tokens - {metadata['total_tokens']}")
    print("model:")
    print(response.response_metadata["model"])
    print()
    print("------ Evaluation -----")
    eval_response, metadata, raw_response = sfa.evaluate_relevance(question, content)
    print("Evaluation:")
    print(eval_response)
    print()
    print("metadata:")
    print(f"Input token - {metadata['input_tokens']}, Output tokens - {metadata['output_tokens']}, Total tokens - {metadata['total_tokens']}")
    """  