# --- แก้ปัญหา sqlite3 version สำหรับ Streamlit Cloud ---
# TODO: ถ้ารันในเครื่อง localhost ให้ comment 3 บรรทัดนี้ออก
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
import requests

# ตั้งค่าหน้า Streamlit
st.set_page_config(page_title="LockLearn lifecoach", page_icon="🧠")

# โหลดฐานข้อมูล ChromaDB
db_path = "./chromadb_database_v2"
client = chromadb.PersistentClient(path=db_path)
collection = client.get_collection(name="recommendations")

# โหลด embedding model
embedding_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

# 🔁 ฟังก์ชันเรียก LLM ผ่าน Together API (chat-completions style)
def query_llm_with_chat(prompt, api_key):
    url = "https://api.together.xyz/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "meta-llama/llama-4-scout-17b-16e-instruct",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 512
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=15)
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
        else:
            return f"❌ API Error {response.status_code}: {response.text}"
    except Exception as e:
        return f"❌ Request failed: {e}"

# 🔍 ดึงคำแนะนำจาก ChromaDB
def retrieve_recommendations(question_embedding, top_k=3):
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k
    )
    if results and 'documents' in results and results['documents']:
        return results['documents'][0]
    return []

# --- UI ---
st.title("🧠 LockLearn AI Chatbot")
st.markdown("Ask about learning, motivation, or self-improvement. Get tailored, encouraging advice 💡")

# ใช้ API key จาก secrets
api_key = st.secrets["TOGETHER_API_KEY"]

user_question = st.text_area("💬 What would you like help with today?")

if st.button("Ask"):
    if not user_question.strip():
        st.warning("⚠️ Please enter your question first.")
    else:
        with st.spinner("Processing..."):
            # สร้าง embedding
            question_embedding = embedding_model.encode(user_question).tolist()

            # ดึงคำแนะนำจาก ChromaDB
            recommendations = retrieve_recommendations(question_embedding, top_k=10)

            # สร้าง prompt แบบ life coach
            prompt = (
                f"Question: {user_question}\n"
                f"Recommendations:\n"
            )
            for rec in recommendations:
                prompt += f"- {rec}\n"

            prompt += """

Please generate a supportive, practical, and encouraging response based on the suggestions above.
Respond in the **same language** as the user's question:
- Thai if the question is in Thai.
- English if the question is in English.

Make your answer concise and natural, like a caring life coach giving motivation in just 1-3 sentences. Keep it positive and uplifting.
"""

            # เรียก LLM
            answer = query_llm_with_chat(prompt, api_key)

            # แสดงผลลัพธ์
            st.markdown("### 🤖 Answer:")
            st.write(answer)

# ปุ่ม clear
if st.button("Clear"):
    st.experimental_rerun()
