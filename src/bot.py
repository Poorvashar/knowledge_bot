# src/bot.py

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dotenv import load_dotenv
load_dotenv()

from slack_bolt import App
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from src.rag.chain import build_rag_chain

# ── Initialization ─────────────────────────────

# Slack App
app = App(
    token=os.environ["SLACK_BOT_TOKEN"],
    signing_secret=os.environ["SLACK_SIGNING_SECRET"]
)

# RAG Chain
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)
chain = build_rag_chain(vectorstore)

# ── Event Handlers ─────────────────────────────

@app.event("app_mention")
def handle_mention(event, say):
    """Called when someone mentions @damon-bot"""
    
    # Strip the mention part and extract just the question
    # e.g., "<@U1234> How do I launch EC2?" → "How do I launch EC2?"
    question = event["text"].split(">", 1)[-1].strip()
    
    if not question:
        say(
            text="What would you like to know? Please include a question with your mention 🤖",
            thread_ts=event["ts"]
        )
        return
    
    try:
        # Generate answer with RAG chain (new API - returns just the answer string)
        answer = chain.invoke(question)
        
        # Reply in the Slack thread
        say(text=answer, thread_ts=event["ts"])
        
    except Exception as e:
        say(
            text=f"⚠️ An error occurred: {str(e)}",
            thread_ts=event["ts"]
        )

# ── Start ─────────────────────────────────────

if __name__ == "__main__":
    print("🤖 Starting Damon Bot...")
    app.start(port=3000)