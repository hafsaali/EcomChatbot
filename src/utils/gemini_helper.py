import json
import google.generativeai as genai
from typing import Dict, List
from config.api_keys import GEMINI_API_KEY
from pathlib import Path
import re

# Load prompt template
PROMPT_TEMPLATE = json.loads(Path("prompts/ecom_prompt.json").read_text())
model = genai.GenerativeModel('gemini-2.0-flash')
genai.configure(api_key=GEMINI_API_KEY)

def format_prompt(query: str, products: List[Dict]) -> str:
    """Inject products into the JSON prompt template."""
    prompt = PROMPT_TEMPLATE.copy()
    prompt["query"] = query
    prompt["products"] = products[:3]  # Limit to top 3
    return json.dumps(prompt, indent=2)

def clean_json_text(text: str) -> str:
    # Remove markdown-style code fences like ```json ... ```
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE | re.MULTILINE)



def generate_response(query: str, products: List[Dict]) -> Dict:
    model = genai.GenerativeModel('gemini-2.0-flash')
    prompt = format_prompt(query, products)

    response = model.generate_content(prompt)
    raw_text = response.text

    if not raw_text:
        raise ValueError("Gemini returned an empty response.")

    cleaned_text = clean_json_text(raw_text)

    try:
        return json.loads(cleaned_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON from Gemini after cleaning:\n{cleaned_text}") from e