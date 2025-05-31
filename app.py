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

# ดึง Together API key จาก secrets หรือ input
if "TOGETHER_API_KEY" in st.secrets:
    api_key = st.secrets["TOGETHER_API_KEY"]
else:
    api_key = st.text_input("Enter your Together API Key", type="password")

# เก็บประวัติแชทใน session_state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def submit():
    user_question = st.session_state.user_input.strip()
    if not user_question or not api_key:
        return
    # embedding
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
    # เรียก LLM
    with st.spinner("Processing..."):
        answer = query_llm_together_api(prompt, api_key)
    # เก็บแชท
    st.session_state.chat_history.append({"user": user_question, "bot": answer})
    # เคลียร์ input
    st.session_state.user_input = ""

# ช่อง input fixed bottom, กด Enter ส่ง
st.text_input(
    label="Ask me something about learning, motivation, or self-improvement:",
    key="user_input",
    placeholder="Type your question and press Enter",
    on_change=submit,
)

# แสดงแชทด้านบน
for chat in st.session_state.chat_history:
    st.markdown(f"**You:** {chat['user']}")
    st.markdown(f"**Bot:** {chat['bot']}")
    st.markdown("---")

# CSS fix input แถบพิมพ์ติดล่าง + ป้องกัน content บัง input
st.markdown("""
<style>
    .stTextInput > div {
        position: fixed !important;
        bottom: 0;
        left: 0;
        width: 100% !important;
        background: white;
        padding: 10px 20px;
        box-shadow: 0 -2px 10px rgba(0,0,0,0.1);
        z-index: 1000;
    }
    .block-container {
        padding-bottom: 70px;
    }
</style>
""", unsafe_allow_html=True)
