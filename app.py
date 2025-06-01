# --- sqlite3 patch สำหรับ Streamlit Cloud ---
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
import requests
import re

# ✅ ตั้งค่าหน้า Streamlit
st.set_page_config(page_title="LockLearn Lifecoach", page_icon="🧠", layout="centered")

# ✅ โหลด ChromaDB
db_path = "./chromadb_database_v2"
client = chromadb.PersistentClient(path=db_path)
collection = client.get_collection(name="recommendations")

# ✅ โหลด embedding model
embedding_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

# ✅ โหลด API Key
api_key = st.secrets["TOGETHER_API_KEY"]

# ✅ ฟังก์ชันเรียก LLM ผ่าน Together API (chat-completions)
def query_llm_with_chat(prompt, api_key):
    url = "https://api.together.xyz/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "meta-llama/llama-4-scout-17b-16e-instruct",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 512
    }
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=15)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"].strip()
        else:
            return f"❌ API Error {response.status_code}: {response.text}"
    except Exception as e:
        return f"❌ Request failed: {e}"

# ✅ ดึงคำแนะนำ
def retrieve_recommendations(question_embedding, top_k=10):
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k
    )
    if results and results.get('documents'):
        return results['documents'][0]
    return []

# ✅ ตรวจว่าข้อความเป็นการปิดบทสนทนาไหม
def is_closing_message(text):
    closing_patterns = [
        r"^ขอบคุณ.*", r"^ขอบใจ.*", r"^โอเค.*", r"^เข้าใจ.*", r"^ได้เลย.*", r"^รับทราบ.*",
        r"^thank(s| you).*", r"^ok.*", r"^got it.*", r"^noted.*", r"^understood.*"
    ]
    text = text.strip().lower()
    if len(text.split()) <= 5:
        for pattern in closing_patterns:
            if re.match(pattern, text):
                return True
    return False

# ✅ สร้าง session state สำหรับเก็บประวัติแชท
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ✅ แสดงประวัติแชท
st.title("🧠 LockLearn Lifecoach")
for entry in st.session_state.chat_history:
    with st.chat_message(entry["role"]):
        st.markdown(entry["content"])

# ✅ ช่องพิมพ์ข้อความด้านล่าง
user_input = st.chat_input("Ask me anything about motivation, study, or self-growth...")

if user_input:
    # บันทึกคำถามของผู้ใช้
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # เช็กว่าควรตอบหรือไม่
    if is_closing_message(user_input):
        reply = "😊 ยินดีเสมอครับ หากต้องการคำแนะนำเพิ่มเติมสามารถถามได้ตลอดเลยนะครับ!"
    else:
        # เรียก LLM พร้อม RAG
        with st.spinner("Thinking..."):
            question_embedding = embedding_model.encode(user_input).tolist()
            recommendations = retrieve_recommendations(question_embedding, top_k=10)

            prompt = f"Question: {user_input}\nRecommendations:\n"
            for rec in recommendations:
                prompt += f"- {rec}\n"

            prompt += """
Please generate a supportive, practical, and encouraging response based on the suggestions above.
Respond in the same language as the user's question:
- Thai if the question is in Thai.
- English if the question is in English.

Make your answer concise and natural, like a caring life coach giving motivation in just 2-3 sentences. Keep it positive and uplifting.
"""
            reply = query_llm_with_chat(prompt, api_key)

    # แสดงคำตอบบอท
    st.session_state.chat_history.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.markdown(reply)
