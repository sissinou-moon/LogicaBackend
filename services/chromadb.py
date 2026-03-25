import chromadb

client = chromadb.CloudClient(
  api_key='ck-t3nTNaVrLJsyZPgFE1X4ZeXg492d6fc9dMV2ATtFnLG',
  tenant='f05fcb00-3bec-4eaf-82a3-915dd5a86bfd',
  database='pfp'
)

collection = client.get_or_create_collection(name="files_memory")

async def add_file_to_memory(path: str, content: str):
    collection.add(
        documents=[content],
        metadatas=[{"path": path}],
        ids=[path]
    )

async def modify_file_in_memory(path: str, content: str):
    collection.update(
        documents=[content],
        metadatas=[{"path": path}],
        ids=[path]
    )

async def get_file_from_memory(path: str):
    return collection.get(
        ids=[path]
    )

async def search_chroma(query: str):
    results = collection.query(
        query_texts=[query],
        n_results=3
    )
    return results