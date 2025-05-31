# --- แก้ปัญหา sqlite3 version สำหรับ Streamlit Cloud ---
# TODO: ถ้ารันในเครื่อง localhost ให้ comment 3 บรรทัดนี้ออก
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
import requests

# ตั้ง config หน้าต่าง app
st.set_page_config(page_title="LockLearn lifecoach", page_icon="🧠")

# โหลดฐานข้อมูล ChromaDB
db_path = "./chromadb_database_v2"  # เปลี่ยนตาม path จริง
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
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()  # ถ้าไม่ 200 จะโยน exception
        data = response.json()
        return data["output"]["choices"][0]["text"].strip()
    except requests.exceptions.HTTPError as http_err:
        return f"❌ HTTP error occurred: {http_err} (URL: {url})"
    except requests.exceptions.Timeout:
        return "❌ Request timed out."
    except requests.exceptions.RequestException as e:
        return f"❌ Request error: {e}"
    except Exception as e:
        return f"❌ Unexpected error: {e}"

# --- UI ---

st.title("🧠 LockLearn AI Chatbot")

# ดึง Together API key จาก secrets หรือ input
if "TOGETHER_API_KEY" in st.secrets:
    api_key = st.secrets["TOGETHER_API_KEY"]
else:
    api_key = st.text_input("Enter your Together API Key", type="password")

# สร้าง session state สำหรับเก็บข้อความคุย
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# UI แสดง chat log
for i, chat in enumerate(st.session_state.chat_history):
    if chat["role"] == "user":
        st.markdown(f"**You:** {chat['content']}")
    else:
        st.markdown(f"**Bot:** {chat['content']}")

# ช่องกรอกข้อความ แบบ input text box อยู่ข้างล่าง พร้อมกด Enter ส่ง
def submit():
    user_question = st.session_state.user_input.strip()
    if user_question == "":
        return
    st.session_state.chat_history.append({"role": "user", "content": user_question})

    # สร้าง embedding และดึงคำแนะนำ
    question_embedding = embedding_model.encode(user_question).tolist()
    recommendations = retrieve_recommendations(question_embedding, top_k=3)

    # สร้าง prompt ให้ LLM
    prompt = f"User question: {user_question}\n\nRelevant recommendations:\n"
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            prompt += f"{i}. {rec}\n"
    else:
        prompt += "No relevant recommendations found.\n"
    prompt += "\nPlease answer the user question using the above recommendations with encouragement and advice."

    with st.spinner("Bot is thinking..."):
        answer = query_llm_together_api(prompt, api_key)

    st.session_state.chat_history.append({"role": "bot", "content": answer})

    # เคลียร์ input text
    st.session_state.user_input = ""

# input text box อยู่ข้างล่างสุด กด Enter ส่งข้อความ
st.text_input("Type your question here and press Enter", key="user_input", on_change=submit)

# scroll to bottom เพื่อให้แถบ input อยู่ล่างสุด (อาจใช้ st.markdown แบบบางครั้งช่วย)
st.markdown("<style>div[data-testid='stVerticalBlock'] > div {max-height: 70vh; overflow-y: auto;}</style>", unsafe_allow_html=True)
