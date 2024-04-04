from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from openai import OpenAI
import os
from dotenv import load_dotenv



load_dotenv('.env')
API_KEY = os.getenv("OPENAI_API_KEY")

client=OpenAI(api_key= API_KEY )
template = """You are provided with a resume {question}.Just do somme technical NER and act like noob make little mistakes okay.Do NER and give answers in the following format.
NER- type

Try to give summary in 3-4 lines only
"""

prompt = PromptTemplate.from_template(template)
prompt = PromptTemplate(input_variables=["question"],
                            template=template)


llm = ChatOpenAI(model="gpt-3.5-turbo",temperature=0.0,openai_api_key=API_KEY)
llm_chain = LLMChain(prompt=prompt, llm=llm)

def callner(text):
    input_text = text 
    if input_text is None:
        raise ValueError("Input text not found")
    question = input_text
    response = llm_chain.run(question)
    return response
