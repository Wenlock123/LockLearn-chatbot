import streamlit as st

# TODO: If trying this app locally, comment out these 3 lines
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb
import requests
import json
from sentence_transformers import SentenceTransformer

# Set Streamlit page config - ต้องเป็นคำสั่งแรก ๆ หลัง import streamlit
st.set_page_config(page_title="LockLearn AI Chatbot", page_icon="🧠")

# โหลด embedding model
@st.cache_resource(show_spinner=False)
def load_embedding_model():
    return SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

embedding_model = load_embedding_model()

# เชื่อมต่อ chromadb persistent client
@st.cache_resource(show_spinner=False)
def get_chromadb_collection():
    client = chromadb.PersistentClient(path="chromadb_database_v2")
    return client.get_collection(name="recommendations")

collection = get_chromadb_collection()

# ฟังก์ชันดึงคำแนะนำด้วย RAG
def retrieve_recommendations(question_embedding, top_k=3):
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k
    )
    if results['documents']:
        return results['documents'][0]
    return []

# ฟังก์ชันเรียก LLM ผ่าน Together API
def query_llm_together_api(prompt, api_key):
    url = "https://api.together.xyz/openai/generate"
    headers = {"Authorization": f"Bearer {api_key}"}
    data = {
        "model": "meta-llama/llama-4-scout-17b-16e-instruct",
        "inputs": prompt,
        "parameters": {
            "temperature": 0.7,
            "max_new_tokens": 512,
            "top_k": 50,
            "top_p": 0.95
        }
    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        return response.json()["output"]["choices"][0]["text"].strip()
    else:
        return f"❌ Failed to get response from LLM: {response.text}"

# UI Streamlit
st.title("🧠 LockLearn AI Chatbot with RAG")

api_key = st.text_input("Enter your Together API Key", type="password")

user_question = st.text_input("Ask your question here:")

if st.button("Ask") and user_question and api_key:
    with st.spinner("Processing your question..."):
        # แปลงข้อความเป็น embedding
        question_embedding = embedding_model.encode(user_question).tolist()

        # ดึงคำแนะนำใกล้เคียงจาก ChromaDB
        recommendations = retrieve_recommendations(question_embedding, top_k=3)

        # สร้าง prompt ให้ LLM รวมคำแนะนำและคำถาม
        prompt = f"Question: {user_question}\n\nRelevant recommendations:\n"
        for i, rec in enumerate(recommendations, 1):
            prompt += f"{i}. {rec}\n"
        prompt += "\nPlease provide a helpful, encouraging, and context-aware answer."

        # เรียก LLM ผ่าน Together API
        answer = query_llm_together_api(prompt, api_key)

        st.markdown("### Answer from AI:")
        st.write(answer)
