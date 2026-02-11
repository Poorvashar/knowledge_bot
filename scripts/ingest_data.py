# scripts/ingest_data.py

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dotenv import load_dotenv
load_dotenv()

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from src.data_sources.google_drive_fetch import list_files_in_folder, download_file_content
from src.rag.ingest import create_chunks, build_vectorstore

FOLDER_ID = "1WhWso5U_WLSyMFU8z0g0ZqpoHWWj1LjT"

# 1. Fetch files from Google Drive
files = list_files_in_folder(FOLDER_ID)

# 2. Download each file's content and wrap it in a Document object
documents = []
for f in files:
    try:
        content = download_file_content(f["id"], f["mimeType"])
        doc = Document(
            page_content=content,
            metadata={"source": f["name"], "file_id": f["id"]}
        )
        documents.append(doc)
        print(f"  ✅ {f['name']}")
    except Exception as e:
        print(f"  ❌ {f['name']}: {e}")

# 3. Chunk the documents
chunks = create_chunks(documents)

# 4. Save to Vector DB
vectorstore = build_vectorstore(chunks)

print("\n🎉 Data ingestion complete!")