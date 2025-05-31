# app.py

import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
import requests

# ตั้งค่า page config
st.set_page_config(page_title="LockLearn AI Chatbot", page_icon="🧠")

# โหลดฐานข้อมูล ChromaDB
db_path = "./chromadb_database_v2"
client = chromadb.PersistentClient(path=db_path)
collection = client.get_collection(name="recommendations")

# โหลด embedding model
embedding_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

def retrieve_recommendations(question_embedding, top_k=3):
    results = collection.query(query_embeddings=[question_embedding], n_results=top_k)
    if results and 'documents' in results and len(results['documents']) > 0:
        return results['documents'][0]
    return []

def query_llm_together_api(prompt, api_key):
    # ตรวจสอบ API endpoint ล่าสุดของ Together.ai
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
        response.raise_for_status()
        json_resp = response.json()
        # ตรวจสอบโครงสร้าง response จริง
        # ตามเอกสาร Together API ปกติ output อยู่ใน json_resp['results'][0]['text'] หรือคล้ายกัน
        if "results" in json_resp and len(json_resp["results"]) > 0:
            return json_resp["results"][0].get("text", "").strip()
        # fallback
        elif "output" in json_resp and "choices" in json_resp["output"]:
            return json_resp["output"]["choices"][0]["text"].strip()
        else:
            return f"❌ Unexpected response format: {json_resp}"
    except requests.exceptions.RequestException as e:
        return f"❌ Request error: {e}"
    except Exception as e:
        return f"❌ Parsing response error: {e}"

# --- UI design ---
st.title("🧠 LockLearn AI Chatbot")

# ดึง api key จาก secrets หรือใส่เอง
api_key = st.secrets.get("TOGETHER_API_KEY", None)
if not api_key:
    api_key = st.text_input("Enter your Together API Key", type="password")

# ตัวแปร session state สำหรับเก็บประวัติแชท
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ช่องข้อความ input ที่อยู่ข้างล่าง
def get_text():
    return st.text_input("Type your question and press Enter", key="input_text")

user_input = get_text()

if user_input and api_key:
    with st.spinner("Generating answer..."):
        # สร้าง embedding
        question_embedding = embedding_model.encode(user_input).tolist()
        # ดึงคำแนะนำ
        recommendations = retrieve_recommendations(question_embedding, top_k=3)

        prompt = f"User question: {user_input}\n\nRelevant recommendations:\n"
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                prompt += f"{i}. {rec}\n"
        else:
            prompt += "No relevant recommendations found.\n"
        prompt += "\nPlease answer the user question using the above recommendations with encouragement and advice."

        answer = query_llm_together_api(prompt, api_key)

        # บันทึกลง chat history
        st.session_state.chat_history.append({"user": user_input, "bot": answer})

    # Clear input field
    st.session_state.input_text = ""

# แสดงแชทย้อนหลัง
for chat in st.session_state.chat_history:
    st.markdown(f"**You:** {chat['user']}")
    st.markdown(f"**Bot:** {chat['bot']}")

# --- Style ให้อยู่ข้างล่าง ---
st.markdown("""
<style>
    /* Fix input box อยู่ล่าง */
    .stTextInput > div > div > input {
        font-size: 16px;
        padding: 12px;
    }
</style>
""", unsafe_allow_html=True)
