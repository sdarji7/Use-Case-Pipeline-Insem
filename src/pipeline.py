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
    model = input("Enter model(deepseek/llama/phi/gemini):").strip().lower()
    prompt_type = "cot"
    system_description = "Below is the problem specification of “NewYork TicketDistributor System”. NewYork Metro station wants to establish a TicketDistributor machine that issues tickets for passengers travelling in metro rails. Travelers have options of selecting a ticket for a single trip, round trips or for multiple trips. They can also issue a metro pass for regular passengers or a time card for a day, a week or a month according to their requirements. The discounts on tickets will be provided to frequent travelling passengers. The machine is also supposed to read the metro pass and time cards issued by the metro counters or machine. The ticket rates differ based on whether the traveler is a child or an adult. The machine is also required to recognize original as well as fake currency notes. The typical transaction consists of a user using the display interface to select the type and quantity of tickets and then choosing a payment method of either cash, credit/debit card or smartcard. The ticket or tickets are printed and dispensed to the user. Also the messaging facilities after every transaction are required on the registered number. The system can also be operated comfortably by a touch-screen. A large number of heavy components are to be used. We do not want our system to slow down, and also usability of the machine. The TicketDistributor must be able to handle several exceptions, such as aborting the transaction for incomplete transactions, insufficient amount given by the travelers to the machine, money return in case of aborted transaction, change return after successful transaction, showing insufficient balance in the card, updated information printed on the tickets e.g. departure time, date, time, price, valid from, valid till, validity duration, ticket issued from and destination station. In case of exceptions, an error message is to be displayed. We do not want user feedback after every development stage but after every two stages to save time. The machine is required to work in a heavy load environment such that at the morning and evening time in weekdays, and in weekends performance and efficiency would not get affected."
    
    result = generate_use_case(model, prompt_type, system_description)
    print(result)
