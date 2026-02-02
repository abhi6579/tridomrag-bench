import os
from dotenv import load_dotenv
from transformers import pipeline

load_dotenv()

class LLMHandler:
    def __init__(self, model="distilgpt2", temperature=0.7):
        """Initialize local LLM"""
        self.model = model
        self.temperature = temperature
        self.generator = pipeline("text-generation", model=self.model)
        print(f"✅ LLM Handler initialized with {model}")
    
    def generate_response(self, prompt, system_message=None, max_tokens=100):
        """Generate response from local model"""
        try:
            response = self.generator(prompt, max_length=max_tokens, temperature=self.temperature)
            return response[0]["generated_text"]
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    def test_connection(self):
        """Test connection"""
        try:
            response = self.generate_response("Recallify is")
            return response
        except Exception as e:
            return f"Connection failed: {str(e)}"

if __name__ == "__main__":
    handler = LLMHandler()
    test_response = handler.test_connection()
    print(f"Test Response: {test_response}")
