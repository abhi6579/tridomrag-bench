"""LLM Handler using Hugging Face transformers (distilgpt2)"""
from transformers import pipeline, set_seed
from config.settings import settings
from core.exceptions import LLMError
from utils.logger import logger

class LLMHandler:
    def __init__(self, model: str = None, temperature: float = None):
        """Initialize Hugging Face LLM pipeline"""
        self.model = model or settings.LLM_MODEL
        self.temperature = temperature or settings.LLM_TEMPERATURE
        
        try:
            # Set seed for reproducibility
            set_seed(42)
            
            # Initialize text generation pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                device=-1  # -1 for CPU, 0 for GPU
            )
            logger.info(f"✅ LLM Handler initialized with {self.model}")
        except Exception as e:
            raise LLMError(f"Failed to initialize LLM: {str(e)}")
    
    def generate_response(self, prompt: str, max_tokens: int = None) -> str:
        """Generate response from LLM using distilgpt2"""
        max_tokens = max_tokens or settings.LLM_MAX_LENGTH
        
        try:
            response = self.pipeline(
                prompt,
                max_length=max_tokens,
                temperature=self.temperature,
                do_sample=True,
                top_p=0.95,
                num_return_sequences=1
            )
            
            # Extract generated text
            generated_text = response[0]["generated_text"]
            
            # Remove the prompt from the generated text
            answer = generated_text[len(prompt):].strip()
            
            return answer if answer else generated_text
        
        except Exception as e:
            logger.error(f"LLM generation error: {str(e)}")
            raise LLMError(f"Failed to generate response: {str(e)}")
    
    def test_connection(self) -> str:
        """Test LLM connection"""
        try:
            prompt = "Recallify is"
            response = self.generate_response(prompt, max_tokens=20)
            logger.info(f"✅ LLM test successful: {response}")
            return response
        except Exception as e:
            error_msg = f"Connection test failed: {str(e)}"
            logger.error(error_msg)
            return error_msg

# Test connection
if __name__ == "__main__":
    handler = LLMHandler()
    test_response = handler.test_connection()
    print(f"Test Response: {test_response}")