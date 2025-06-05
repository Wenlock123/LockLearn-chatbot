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
st.set_page_config(page_title="LockLearn Lifecoach", page_icon="💖", layout="centered")

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

# ✅ ตรวจข้อความมั่วหรือพิมพ์ผิด
def is_gibberish_or_typo(text):
    text = text.strip()
    if len(text) <= 2:
        return True
    words = text.split()
    if len(words) == 1 and not re.search(r'[a-zA-Zก-๙]', words[0]):
        return True
    return False

# ✅ ตรวจจับภาษา (ไทยหรืออังกฤษ)
def detect_language(text):
    thai_chars = re.findall(r'[\u0E00-\u0E7F]', text)
    return "th" if len(thai_chars) / max(len(text), 1) > 0.3 else "en"

# ✅ สร้าง session state สำหรับเก็บประวัติแชท
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ✅ แสดงประวัติแชท
st.title("💖 LockLearn Lifecoach")
for entry in st.session_state.chat_history:
    with st.chat_message(entry["role"]):
        st.markdown(entry["content"])

# ✅ ช่องพิมพ์ข้อความด้านล่าง
user_input = st.chat_input("How can I support you today? Feel free to ask me anything")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    lang = detect_language(user_input)

    if is_gibberish_or_typo(user_input):
        reply = {
            "th": "😅 ผมไม่แน่ใจว่าคุณหมายถึงอะไร ลองพิมพ์ใหม่อีกครั้งนะครับ",
            "en": "😅 I'm not sure what you mean. Could you try rephrasing it?"
        }[lang]
    elif is_closing_message(user_input):
        reply = {
            "th": "😊 ยินดีเสมอครับ หากต้องการคำแนะนำเพิ่มเติมสามารถถามได้ตลอดเลยนะครับ!",
            "en": "😊 You're always welcome! Feel free to ask if you need more support!"
        }[lang]
    else:
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

If the user's question is in Thai, respond in a polite and feminine tone using "ค่ะ" at the end of the sentence, as if you are a female life coach giving kind, warm, and caring motivation.
If the user's question is in English, keep a kind and uplifting tone like a supportive female life coach.

Keep the response concise (1–2 sentences), natural, and motivating.
"""

            reply = query_llm_with_chat(prompt, api_key)

    st.session_state.chat_history.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.markdown(reply)
