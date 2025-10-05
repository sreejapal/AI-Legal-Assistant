import os
from dotenv import load_dotenv
import openai

# Load environment variables
load_dotenv()

# Read API key from environment variable
openai.api_key = os.getenv("LLM_API_KEY", "")

if not openai.api_key:
    raise RuntimeError("LLM_API_KEY is not set. Create a .env file or set the environment variable.")

# Example request (for OpenAI >= 1.0.0)
response = openai.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Say hello in one sentence."}]
)

print(response.choices[0].message.content)

import os
from dotenv import load_dotenv
import openai

# Load environment variables from .env if present
load_dotenv()

# Read API key from environment variable
openai.api_key = os.getenv("LLM_API_KEY", "")

if not openai.api_key:
    raise RuntimeError("LLM_API_KEY is not set. Create a .env file or set the environment variable.")

# Example request (for OpenAI >= 1.0.0)
response = openai.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Say hello in one sentence."}]
)

print(response.choices[0].message.content) 
