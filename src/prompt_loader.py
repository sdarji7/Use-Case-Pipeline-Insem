import os

def load_prompt(file_name: str) -> str:
    """Loads a prompt from the prompts directory."""
    prompt_path = os.path.join("prompts", file_name)
    try:
        with open(prompt_path, "r", encoding="utf-8") as file:
            return file.read().strip()
    except FileNotFoundError:
        print(f"Warning: Prompt file {file_name} not found.")
        return ""

# Example usage
if __name__ == "__main__":
    print(load_prompt("zero_shot.txt"))
