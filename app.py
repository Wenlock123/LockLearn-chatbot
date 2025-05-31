import streamlit as st
import json
import requests
import chromadb
from sentence_transformers import SentenceTransformer

# --- SET PAGE CONFIG ต้องอยู่บรรทัดแรกๆ หลัง import streamlit ---
st.set_page_config(page_title="LockLearn AI Chatbot", page_icon="🧠")

# TODO: For Streamlit Cloud with pysqlite3 fix
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


# --- โหลด Embedding Model ---
@st.cache_resource(show_spinner=False)
def load_embedding_model():
    model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
    return model

embedding_model = load_embedding_model()

# --- เชื่อมต่อ ChromaDB ---
@st.cache_resource(show_spinner=False)
def connect_chromadb(db_path):
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_collection(name="recommendations")
    return collection

db_path = 'chromadb_database_v2'  # ปรับ path ตามที่เก็บจริงใน repo หรือ Streamlit Cloud
collection = connect_chromadb(db_path)

# --- ฟังก์ชันดึงคำแนะนำใกล้เคียง (RAG) ---
def retrieve_recommendations(question_embedding, top_k=5):
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k
    )
    if results and results['documents']:
        return results['documents'][0]
    return []

# --- ฟังก์ชันเรียก LLM ผ่าน Together API ---
TOGETHER_API_URL = "https://api.together.xyz/api/v0/models/meta-llama/llama-4-scout-17b-16e-instruct/invoke"
TOGETHER_API_KEY = st.secrets["together_api_key"]  # เก็บ API key ใน Streamlit secrets

def query_together_llm(prompt, max_tokens=512, temperature=0.7):
    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "stop": ["###"]
        }
    }
    response = requests.post(TOGETHER_API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()["output"]["choices"][0]["text"].strip()
    else:
        return f"❌ Failed to get response from LLM: {response.text}"

# --- Streamlit UI ---
st.title("🧠 LockLearn AI Chatbot")

user_question = st.text_input("ถามคำถามของคุณได้เลย:", "")

if user_question:
    with st.spinner("กำลังประมวลผล..."):
        # แปลงคำถามเป็น embedding
        question_embedding = embedding_model.encode(user_question).tolist()

        # ดึงคำแนะนำใกล้เคียงจาก ChromaDB
        recommendations = retrieve_recommendations(question_embedding, top_k=5)

        # สร้าง prompt รวมคำแนะนำ + คำถามสำหรับ LLM
        prompt = "You are a helpful life coach. Use the following recommendations to answer the question.\n\n"
        for idx, rec in enumerate(recommendations, 1):
            prompt += f"Recommendation {idx}: {rec}\n"
        prompt += f"\nQuestion: {user_question}\nAnswer:"

        # เรียก LLM เพื่อสร้างคำตอบ
        answer = query_together_llm(prompt)

        st.markdown("### คำตอบจาก AI:")
        st.write(answer)
