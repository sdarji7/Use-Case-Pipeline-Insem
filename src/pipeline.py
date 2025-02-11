import os
import dotenv
from langchain_community.llms import Ollama
from langchain_google_genai import GoogleGenerativeAI
from prompt_loader import load_prompt

# Load environment variables
dotenv.load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Model Mapping
MODELS = {
    "deepseek": {"type": "ollama", "name": "deepseek-r1:1.5b"},
    "llama": {"type": "ollama", "name": "llama3.2:3b"},
    "phi": {"type": "ollama", "name": "phi3:3.8b"},
    "gemini": {"type": "gemini", "name": "gemini-pro"}
}

def generate_use_case(model_name: str, prompt_type: str, system_description: str):
    if model_name not in MODELS:
        raise ValueError("Invalid model name")
    
    # Load prompt template
    # print(prompt_type)
    prompt_template = load_prompt(f"{prompt_type}.txt")
    # print(prompt_template)
    
    # Replace {system_description} placeholder in the prompt template
    full_prompt = prompt_template.replace("{system_description}", system_description)
    
    model_config = MODELS[model_name]

    if model_config["type"] == "ollama":
        # Use LangChain's Ollama wrapper
        llm = Ollama(model=model_config["name"])
        response = llm.invoke(full_prompt)
    
    elif model_config["type"] == "gemini":
        # Use LangChain's Google Gemini wrapper
        llm = GoogleGenerativeAI(model=model_config["name"], google_api_key=GEMINI_API_KEY)
        response = llm.invoke(full_prompt)

    else:
        return "Invalid model type"

    return response

if __name__ == "__main__":
    model = "gemini"  # Example usage
    prompt_type = "zero_shot"
    system_description = "A smart home automation system that manages lighting, temperature, and security."
    
    result = generate_use_case(model, prompt_type, system_description)
    print(result)
