# app.py
# ต้องนำเข้า pysqlite3 แทน sqlite3 ก่อน import chromadb (เพื่อแก้ sqlite3 เวอร์ชันเก่าใน Streamlit Cloud)
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
import requests

# ต้องตั้ง set_page_config เป็นคำสั่งแรกสุด (หลัง import streamlit)
st.set_page_config(page_title="LockLearn AI Chatbot", page_icon="🧠")

# โหลดฐานข้อมูล ChromaDB
db_path = "./chromadb_database_v2"  # หรือเปลี่ยนเป็น path จริงตาม repo ของคุณ
client = chromadb.PersistentClient(path=db_path)
collection = client.get_collection(name="recommendations")

# โหลด embedding model
embedding_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

# ฟังก์ชันดึงคำแนะนำ (RAG)
def retrieve_recommendations(question_embedding, top_k=3):
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k
    )
    if results['documents']:
        return results['documents'][0]
    return []

# ฟังก์ชันเรียกใช้ LLM ผ่าน Together API
def query_llm_together_api(prompt, api_key):
    url = "https://api.together.xyz/inference/meta-llama/llama-4-scout-17b-16e-instruct"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.95,
        }
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()["output"]["choices"][0]["text"].strip()
    else:
        return f"❌ Failed to get response from LLM: {response.text}"

# Streamlit UI
st.title("🧠 LockLearn AI Chatbot")

api_key = st.text_input("Enter your Together API Key", type="password")
user_question = st.text_area("Ask me something about learning, motivation, or self-improvement:")

if st.button("Ask") and api_key and user_question:
    with st.spinner("Processing..."):
        # Embed user question
        question_embedding = embedding_model.encode(user_question).tolist()
        # ดึงคำแนะนำที่ใกล้เคียง
        recommendations = retrieve_recommendations(question_embedding, top_k=3)
        
        # สร้าง prompt สำหรับ LLM
        prompt = f"User question: {user_question}\n\nRelevant recommendations:\n"
        for i, rec in enumerate(recommendations, 1):
            prompt += f"{i}. {rec}\n"
        prompt += "\nPlease answer the user question using the above recommendations with encouragement and advice."

        # เรียก LLM
        answer = query_llm_together_api(prompt, api_key)
        st.markdown("### 🤖 Answer:")
        st.write(answer)

