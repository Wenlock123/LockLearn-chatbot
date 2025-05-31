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
st.set_page_config(page_title="LockLearn lifecoach", page_icon="🧠", layout="wide")

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

# ฟังก์ชันเรียกใช้ LLM ผ่าน Together API (แก้ URL โมเดลให้ถูกต้อง)
def query_llm_together_api(prompt, api_key):
    url = "https://api.together.xyz/inference/meta-llama/llama-4-scout-17b-16e-instruct"  # URL ตัวใหม่
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
        return f"❌ Request error: {response.status_code} {response.reason} - {response.text}"

# --- ตรวจสอบและเริ่มต้น session_state chat_history ---
if "chat_history" not in st.session_state or not all(isinstance(c, dict) and "role" in c and "content" in c for c in st.session_state.get("chat_history", [])):
    st.session_state.chat_history = []

# ดึง Together API key จาก secrets หรือ input
if "TOGETHER_API_KEY" in st.secrets:
    api_key = st.secrets["TOGETHER_API_KEY"]
else:
    api_key = st.text_input("Enter your Together API Key", type="password")

st.title("🧠 LockLearn AI Chatbot")

# ช่องพิมพ์ข้อความ user แบบ multiline แต่กด Enter ส่งได้ (ด้วย on_change + key)
def submit_question():
    user_question = st.session_state.input_text.strip()
    if not user_question:
        return
    # เก็บข้อความ user ลง chat_history
    st.session_state.chat_history.append({"role": "user", "content": user_question})
    # สร้าง embedding
    question_embedding = embedding_model.encode(user_question).tolist()
    # ดึงคำแนะนำ
    recommendations = retrieve_recommendations(question_embedding, top_k=3)

    # สร้าง prompt ให้ LLM
    prompt = f"User question: {user_question}\n\nRelevant recommendations:\n"
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            prompt += f"{i}. {rec}\n"
    else:
        prompt += "No relevant recommendations found.\n"
    prompt += "\nPlease answer the user question using the above recommendations with encouragement and advice."

    # เรียก LLM ผ่าน Together API
    with st.spinner("Thinking..."):
        answer = query_llm_together_api(prompt, api_key)

    # เก็บคำตอบ bot ลง chat_history
    st.session_state.chat_history.append({"role": "bot", "content": answer})

    # เคลียร์ input text
    st.session_state.input_text = ""

# แสดง chat history
chat_container = st.container()
with chat_container:
    for chat in st.session_state.chat_history:
        role = chat.get("role", "bot")
        content = chat.get("content", "")
        if role == "user":
            st.markdown(f"<div style='text-align: right; background-color:#DCF8C6; padding:8px; border-radius:10px; margin:5px 0;'>{content}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='text-align: left; background-color:#F1F0F0; padding:8px; border-radius:10px; margin:5px 0;'>{content}</div>", unsafe_allow_html=True)

# แถบพิมพ์ข้อความอยู่ล่างสุด
input_col1, input_col2 = st.columns([8,1])
with input_col1:
    user_input = st.text_area(
        label="",
        key="input_text",
        height=70,
        placeholder="Type your message here and press Enter...",
        on_change=submit_question,
        args=(),
        help="Press Enter to send",
    )

with input_col2:
    send_button = st.button("Send", on_click=submit_question)

# ทำให้กด Enter ส่งได้จริง (ใช้ hack เล็กน้อย)
# โดยในช่อง text_area ถ้กด Enter จะมี '\n' ถ้าเรากด Shift+Enter จะขึ้น 2 ตัว \n
# แต่ Streamlit ยังไม่มี native support กด Enter ส่ง เลยใช้ on_change แทน

# --- ปิด spinner หลังคิดเสร็จแล้ว ไม่เทาหน้าจอ ---

