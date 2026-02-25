from dotenv import load_dotenv
load_dotenv()

import os
import httpx
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY", "").strip(),
    http_client=httpx.Client(verify=False, timeout=30.0)  # TEMP BYPASS
)

resp = client.responses.create(
    model="gpt-4o-mini",
    input="Reply with exactly: OpenAI API working",
    max_output_tokens=20,
)

print(getattr(resp, "output_text", None))