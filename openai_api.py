from openai import OpenAI
import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Debugging: check if key is loaded
print("Loaded GROQ_API_KEY:", os.getenv("GROQ_API_KEY"))

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

response = client.responses.create(
    model="openai/gpt-oss-120B",
    input="Explain the importance of fast language models",
)

print(response.output_text)

