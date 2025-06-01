# app.py

# --- แก้ปัญหา sqlite3 version สำหรับ Streamlit Cloud ---
# TODO: ถ้ารันในเครื่อง localhost ให้ comment 3 บรรทัดนี้ออก
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
import requests

# ต้องตั้ง set_page_config เป็นคำสั่งแรกสุดหลัง import streamlit
st.set_page_config(page_title="LockLearn lifecoach", page_icon="🧠")

# โหลดฐานข้อมูล ChromaDB
db_path = "./chromadb_database_v2"  # เปลี่ยน path ตามจริงในระบบคุณ
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
    if results and 'documents' in results and len(results['documents']) > 0:
        return results['documents'][0]
    return []

# ฟังก์ชันเรียกใช้ LLM ผ่าน Together API (เวอร์ชันใหม่)
def query_llm_together_api(prompt, api_key):
    url = "https://api.together.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "meta-llama/llama-4-scout-17b-16e-instruct",
        "messages": [
            {"role": "system", "content": "You are a kind and encouraging life coach. Keep answers short, supportive, and motivational."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.95
    }
    response = requests.post(url, headers=headers, json=payload)

    if response.status_code == 200:
        try:
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            return f"❌ Error parsing response: {e}"
    else:
        return f"❌ Failed to get response from LLM: Status {response.status_code} - {response.text[:300]}"

# --- UI ---
st.title("🧠 LockLearn AI Chatbot")

# รับ API Key (หากไม่ได้ใช้ secrets)
if "TOGETHER_API_KEY" in st.secrets:
    api_key = st.secrets["TOGETHER_API_KEY"]
else:
    api_key = st.text_input("Enter your Together API Key", type="password")

user_question = st.text_area("Ask me something about learning, motivation, or self-improvement:")

if st.button("Ask") and api_key and user_question:
    with st.spinner("Processing..."):
        # สร้าง embedding
        question_embedding = embedding_model.encode(user_question).tolist()
        
        # ดึงคำแนะนำ
        recommendations = retrieve_recommendations(question_embedding, top_k=3)

        # สร้าง prompt สำหรับ LLM
        prompt = (
            "You are a life coach. Answer the user question clearly in 1–3 sentences "
            "using the relevant recommendations below. Always give encouragement and positive advice.\n\n"
            f"User question: {user_question}\n\nRelevant recommendations:\n"
        )
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                prompt += f"{i}. {rec}\n"
        else:
            prompt += "No relevant recommendations found.\n"
        prompt += "\nAnswer shortly and kindly:"

        # เรียก LLM ผ่าน Together API
        answer = query_llm_together_api(prompt, api_key)
        st.markdown("### 🤖 Answer:")
        st.write(answer)
