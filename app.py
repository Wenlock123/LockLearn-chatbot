# app.py

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
import requests

st.set_page_config(page_title="LockLearn AI", page_icon="🧠")

# โหลดฐานข้อมูล ChromaDB
client = chromadb.PersistentClient(path="./chromadb_database_v2")
collection = client.get_collection(name="recommendations")

# โหลด embedding model
embedding_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

# ฟังก์ชันดึงคำแนะนำ
def retrieve_recommendations(question_embedding, top_k=3):
    results = collection.query(query_embeddings=[question_embedding], n_results=top_k)
    if results and 'documents' in results and len(results['documents']) > 0:
        return results['documents'][0]
    return []

# ฟังก์ชันเรียก Together AI Chat API
def query_llm_together_chat(prompt, api_key):
    url = "https://api.together.xyz/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a warm, supportive, and practical life coach. "
                    "When answering questions, keep your replies short and clear — no more than 1–3 sentences. "
                    "Speak like a real human, not like a chatbot. "
                    "Always end your reply with an encouraging message."
                )
            },
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.95,
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        try:
            return response.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            return f"❌ Error parsing response: {e}"
    else:
        return f"❌ Request error: {response.status_code} - {response.reason}\n{response.text}"

# เริ่มต้น session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("🧠 LockLearn Chatbot")

# ดึง API Key
if "TOGETHER_API_KEY" in st.secrets:
    api_key = st.secrets["TOGETHER_API_KEY"]
else:
    api_key = st.text_input("Enter Together API Key", type="password")

# กล่องพิมพ์ข้อความ
user_input = st.text_input("Ask something about learning, motivation, or self-improvement:")

# ส่งคำถาม
if st.button("Send") and user_input and api_key:
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    with st.spinner("Thinking..."):
        embedding = embedding_model.encode(user_input).tolist()
        recommendations = retrieve_recommendations(embedding, top_k=3)

        prompt = f"User question: {user_input}\n\nRelevant recommendations:\n"
        prompt += "\n".join([f"{i+1}. {rec}" for i, rec in enumerate(recommendations)])
        prompt += "\n\nAnswer as a helpful life coach."

        answer = query_llm_together_chat(prompt, api_key)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})

# แสดงประวัติแชท
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['content']}")
    else:
        st.markdown(f"**Bot:** {msg['content']}")
