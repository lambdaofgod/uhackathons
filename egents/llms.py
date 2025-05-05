import pathlib
import os

# Load API keys from files
def load_api_keys():
    api_key_path = "~/.keys/anthropic_key.txt"
    with open(pathlib.Path(api_key_path).expanduser()) as f:
        api_key = f.read().strip()
        os.environ["ANTHROPIC_API_KEY"] = api_key

    with open(pathlib.Path("~/.keys/gemini_api_key.txt").expanduser()) as f:
        api_key = f.read().strip()
        os.environ["GEMINI_API_KEY"] = api_key

    with open(pathlib.Path("~/.keys/brave.txt").expanduser()) as f:
        api_key = f.read().strip()
        os.environ["BRAVE_API_KEY"] = api_key

# Model names
anthropic_model_name = "anthropic/claude-3-7-sonnet-20250219"
gemini_model_name = "gemini/gemini-2.0-flash-thinking-exp"
