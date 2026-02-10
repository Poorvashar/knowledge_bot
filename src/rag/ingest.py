from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# Chunking
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=200,
    separators=["\n\n", "\n", ".", " "]
)
chunks = splitter.split_documents(documents)

# Embedding + Vector DB
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(
    chunks, embeddings, persist_directory="./chroma_db"
)