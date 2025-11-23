import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

class LLMModel:
    def __init__(self, model_name="openai/gpt-oss-120b"):
        if not model_name:
            raise ValueError("Model is not defined.")

        self.model_name = model_name
        self.client = ChatGroq(
            model=self.model_name,
            groq_api_key=os.getenv("GROQ_API_KEY")
        )
        
    def get_model(self):
        return self.client

if __name__ == "__main__":
    llm_instance = LLMModel()  
    llm_model = llm_instance.get_model()
    
    response = llm_model.invoke("hi")
    print(response)
