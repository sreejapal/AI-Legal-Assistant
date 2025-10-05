import os
from dotenv import load_dotenv
import openai

# Load .env
load_dotenv()
openai.api_key = os.getenv("LLM_API_KEY", "")

if not openai.api_key:
    raise RuntimeError("LLM_API_KEY is not set. Create a .env file or set the environment variable.")

messages = [
    {"role": "user", "content": "Say hello in one sentence."}
]

try:
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )

    reply = response.choices[0].message.content
    print("OpenAI says:", reply)

except openai.error.AuthenticationError:
    print("Invalid API key. Check your key and try again.")
except openai.error.APIConnectionError:
    print("Could not connect to OpenAI API. Check your internet connection.")
except Exception as e:
    print("Something went wrong:", str(e))

import os
from dotenv import load_dotenv
import openai

# Load .env
load_dotenv()
openai.api_key = os.getenv("LLM_API_KEY", "")

if not openai.api_key:
    raise RuntimeError("LLM_API_KEY is not set. Create a .env file or set the environment variable.")

messages = [
    {"role": "user", "content": "Say hello in one sentence."}
]

try:
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )

    reply = response.choices[0].message.content
    print("OpenAI says:", reply)

except openai.error.AuthenticationError:
    print("Invalid API key. Check your key and try again.")
except openai.error.APIConnectionError:
    print("Could not connect to OpenAI API. Check your internet connection.")
except Exception as e:
    print("Something went wrong:", str(e))
