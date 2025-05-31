import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
import requests
import os

# โหลดโมเดล embed
@st.cache_resource(show_spinner=True)
def load_embedding_model():
    return SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

model = load_embedding_model()

# เชื่อมกับ ChromaDB (แก้ path ให้ตรงกับของคุณ)
DB_PATH = '/content/drive/MyDrive/LockLearn/chromadb_database_v2'  # เปลี่ยนตามจริง
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_collection(name="recommendations")

# ฟังก์ชันดึงคำแนะนำใกล้เคียงจาก embedding
def retrieve_recommendations(question_embedding, top_k=3):
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k,
        include=['documents']  # ให้ดึงเอกสารที่เป็นคำแนะนำ
    )
    if results and results['documents']:
        return results['documents'][0]
    return []

# ฟังก์ชันเรียก LLM (ตัวอย่าง: Together API)
def call_llm(prompt: str) -> str:
    api_key = os.getenv("TOGETHER_API_KEY")  # ใส่ API key ใน env vars
    if not api_key:
        return "Error: API key not set."
    
    url = "https://api.together.xyz/v3/llm/meta-llama/llama-4-scout-17b-16e-instruct/generate"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "prompt": prompt,
        "max_new_tokens": 256,
        "temperature": 0.7,
        "stop": ["\n\n"]
    }
    resp = requests.post(url, json=data, headers=headers)
    if resp.status_code == 200:
        return resp.json().get("results", [{}])[0].get("text", "No response")
    else:
        return f"Error from LLM API: {resp.status_code} {resp.text}"

# UI
st.title("LockLearn Life Coach Chatbot")

user_question = st.text_input("ถามอะไรมาได้เลย:")

if user_question:
    # สร้าง embedding คำถาม
    question_embedding = model.encode(user_question).tolist()

    # ดึงคำแนะนำจาก ChromaDB
    recs = retrieve_recommendations(question_embedding, top_k=3)

    if recs:
        context = "\n".join(f"- {r}" for r in recs)
    else:
        context = "No relevant advice found."

    # สร้าง prompt สำหรับ LLM
    prompt = (
        f"You are a friendly and empathetic life coach. "
        f"Use the following advice to help the user with their question.\n\n"
        f"Advice:\n{context}\n\n"
        f"User question: {user_question}\n"
        f"Answer briefly with 1-3 sentences, encouraging and human-like."
    )

    # เรียก LLM เพื่อสร้างคำตอบ
    answer = call_llm(prompt)

    st.markdown("### คำแนะนำจากระบบ:")
    st.write(answer)
