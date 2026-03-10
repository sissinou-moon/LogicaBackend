import google.generativeai as genai
import os
import json

genai.configure(api_key="AIzaSyCQrckUSrppTYxf7e6w2QO2fPqm8LieR-o")

model = genai.GenerativeModel("gemma-3-27b-it")

async def askAI(message: str):
    prompt = f"""
You are an AI file manager.

Always respond ONLY with valid JSON.

When generating markdown:
- Use double line breaks (\n\n) between paragraphs
- Use '-' for bullet lists, '##' for headings
- Bold important words with '**'
Always return clean markdown content.

JSON format:
{{
  "action": "create_file | delete_file | rename_file | list_files | answer",
  "path": "file path or folder path",
  "content": "markdown file content",
  "message": "response to the user"
}}

User request:
{message}
"""
    response = model.generate_content(prompt)
    text = response.text.strip()

    # remove markdown if Gemini adds it
    if text.startswith("```"):
        text = text.replace("```json", "").replace("```", "").strip()

    data = json.loads(text)

    return data