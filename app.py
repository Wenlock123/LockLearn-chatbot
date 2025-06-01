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
        try:
            return response.json()["output"]["choices"][0]["text"].strip()
        except Exception as e:
            return f"Error parsing LLM response: {e}"
    else:
        return f"❌ Failed to get response from LLM: {response.text}"

# --- UI ---
st.title("🧠 LockLearn AI Chatbot")

# ใช้ API key จาก secrets เท่านั้น (ปลอดภัยกว่า)
api_key = st.secrets["TOGETHER_API_KEY"]

user_question = st.text_area("Ask me something about learning, motivation, or self-improvement:")

if st.button("Ask") and user_question:
    with st.spinner("Processing..."):
        # สร้าง embedding
        question_embedding = embedding_model.encode(user_question).tolist()
        # ดึงคำแนะนำ
        recommendations = retrieve_recommendations(question_embedding, top_k=3)

        # สร้าง prompt
        prompt = f"User question: {user_question}\n\nRelevant recommendations:\n"
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                prompt += f"{i}. {rec}\n"
        else:
            prompt += "No relevant recommendations found.\n"
        prompt += "\nPlease answer the user question using the above recommendations with encouragement and advice."

        # เรียก LLM ผ่าน Together API
        answer = query_llm_together_api(prompt, api_key)
        st.markdown("### 🤖 Answer:")
        st.write(answer)
