#import google.generativeai as genai
import os
import json
from openai import OpenAI

#genai.configure(api_key="AIzaSyCQrckUSrppTYxf7e6w2QO2fPqm8LieR-o")

#model = genai.GenerativeModel("gemma-3-27b-it")

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key="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",
)

async def askAI(message: str):
    prompt = f"""
You are an AI file manager.

Always respond ONLY with valid JSON.

When generating markdown:
- Use '-' for bullet lists, '##' for headings
- Bold important words with '**'
Always return clean markdown content.

JSON format:
{
    [{
        "action": "create_file | create_folder | delete_file | rename_file | list_files | answer | modify_content",
        "path": "file path or folder path",
        "old_path": "old file/folder path (only for rename_file)",
        "new_path": "new file/folder path (only for rename_file)",
        "content": "markdown file content",
        "message": "response to the user"
    }]
}

It can be more than one action, simply return more than one json in the list

User request:
{message}
"""
    completion = client.chat.completions.create(
        model="XiaomiMiMo/MiMo-V2-Flash:novita",
        messages=[
            {
                "role": "system",
                "content": prompt
            },
            {
                "role": "user",
                "content": message
            }
        ],
    )

    raw = completion.choices[0].message.content

    try:
        return json.loads(raw)   # ✅ convert to real JSON
    except:
        return {
            "action": "answer",
            "message": raw,
            "data": None
        }