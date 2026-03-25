#import google.generativeai as genai
from services.chromadb import search_chroma
import os
import json
from openai import OpenAI
import ast

#genai.configure(api_key="AIzaSyCQrckUSrppTYxf7e6w2QO2fPqm8LieR-o")

#model = genai.GenerativeModel("gemma-3-27b-it")

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key="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",
)

async def askAI(message: str, history: list = []):

    search_results = await search_chroma(message) 
    
    # Format the retrieved knowledge
    search_results = await search_chroma(message)

    knowledge_context = ""

    if search_results and "documents" in search_results:
        docs = search_results.get("documents", [[]])[0]
        metas = search_results.get("metadatas", [[]])[0]

        for i in range(min(2, len(docs))):
            path = metas[i].get("path", "") if i < len(metas) else ""
            content = docs[i][:300]

            knowledge_context += f"File: {path}\nContent: {content}\n\n"

    prompt = f"""
You are an AI file manager.

Always respond ONLY with valid JSON.

KNOWLEDGE FROM MEMORY (RAG):
{knowledge_context}


When generating markdown:
- Use '-' for bullet lists, '##' for headings
- Bold important words with '**'
Always return clean markdown content.

INSTRUCTIONS:
- Always respond ONLY with valid JSON.
- Use the provided context to answer questions about file contents.
- If the user asks to "modify" a file, use 'modify_content' action.

JSON format:
[
    {{
        "action": "create_file | create_folder | delete_file | rename_file | list_files | answer | modify_content",
        "path": "file path",
        "content": "content",
        "message": "response to user"
    }}
]

It can be more than one action, simply return more than one json in the list

"""
    # 2. Build the message list including History
    messages = [{"role": "system", "content": prompt}]
    
    # Add previous chat history (Short-term memory)
    for msg in history:
        messages.append(msg)
        
    # Add the current user message
    messages.append({"role": "user", "content": message})

    completion = client.chat.completions.create(
        model="XiaomiMiMo/MiMo-V2-Flash:novita",
        messages=messages,
    )

    raw = completion.choices[0].message.content.strip()

    # remove ``` blocks
    if raw.startswith("```"):
        raw = raw.replace("```json", "").replace("```", "").strip()

    # fix wrapped string
    if raw.startswith('"') and raw.endswith('"'):
        raw = raw[1:-1]

    # TRY JSON
    try:
        return json.loads(raw)
    except:
        try:
            # fallback: handle Python-like list
            return ast.literal_eval(raw)
        except:
            return [{
                "action": "answer",
                "message": raw
        }]