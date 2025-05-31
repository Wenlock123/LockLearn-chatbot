import streamlit as st
import torch
from sentence_transformers import SentenceTransformer
import chromadb
import os
import json
import requests
from typing import List

# ============================ CONFIG ============================

TOGETHER_API_KEY = st.secrets.get("TOGETHER_API_KEY", os.getenv("TOGETHER_API_KEY"))
CHROMA_PATH = "./chromadb_database_v2"  # เปลี่ยนตามที่ mount Google Drive แล้ว
CHROMA_COLLECTION_NAME = "recommendations"

EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
LLM_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# ============================ INITIALIZE ============================

@st.cache_resource
def load_model():
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, device

model, device = load_model()

@st.cache_resource
def load_chroma_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_collection(name=CHROMA_COLLECTION_NAME)

collection = load_chroma_collection()

# ============================ FUNCTION ============================

def embed_text(text: str):
    embedding = model.encode(text, convert_to_tensor=True, device=device)
    return embedding.cpu().numpy().tolist()

def retrieve_docs(embedding: List[float], top_k=5):
    results = collection.query(query_embeddings=[embedding], n_results=top_k)
    return results["documents"][0] if results["documents"] else []

def generate_answer(prompt: str):
    url = "https://api.together.xyz/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful and encouraging Thai education assistant."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 512,
        "temperature": 0.7,
    }

    res = requests.post(url, headers=headers, json=payload)
    res.raise_for_status()
    return res.json()["choices"][0]["message"]["content"]

def build_prompt(question: str, docs: List[str]):
    context = "\n\n".join(docs)
    return f"""คำถามของผู้ใช้: {question}

คำแนะนำที่เกี่ยวข้อง:
{context}

โปรดใช้ข้อมูลข้างต้นเพื่อตอบคำถามของผู้ใช้ พร้อมให้กำลังใจ และคำแนะนำที่เหมาะสม"""

# ============================ UI ============================

st.set_page_config(page_title="LockLearn AI Chatbot", page_icon="💬")
st.title("💬 LockLearn Chatbot")
st.markdown("ถามคำถามของคุณเกี่ยวกับการเรียน หรือการพัฒนาตนเองได้เลย!")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("ถามคำถามเกี่ยวกับการเรียน หรือการพัฒนาตนเองได้เลย...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("กำลังหาคำแนะนำ..."):
            embedding = embed_text(user_input)
            docs = retrieve_docs(embedding)
            prompt = build_prompt(user_input, docs)
            response = generate_answer(prompt)
            st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
