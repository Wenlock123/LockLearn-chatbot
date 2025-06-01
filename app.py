# --- sqlite3 patch สำหรับ Streamlit Cloud ---
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
import requests
import re
from pythainlp.spell import correct
from langdetect import detect

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

# ✅ ตรวจสอบข้อความมั่ว (เช่น พิมพ์สระพยัญชนะมั่ว)
def is_gibberish_or_typo(text):
    words = text.strip().split()
    if len(words) == 0:
        return True
    # ถ้ามีแต่ตัวอักษรไม่เป็นคำ
    if len(text) < 4:
        return True
    thai_chars = sum(1 for ch in text if '\u0E00' <= ch <= '\u0E7F')
    eng_chars = sum(1 for ch in text if ch.isalpha())
    if thai_chars == 0 and eng_chars == 0:
        return True
    return False

# ✅ แก้ไขสะกดผิด (ใช้เฉพาะภาษาไทย)
def autocorrect_text(text):
    try:
        lang = detect(text)
        if lang == 'th':
            corrected = ' '.join([correct(word) for word in text.split()])
            return corrected
        else:
            return text
    except:
        return text

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
    lang = detect(user_input)

    # บันทึกข้อความของผู้ใช้
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    if is_gibberish_or_typo(user_input):
        reply = "😅 ฉันไม่แน่ใจว่าคุณหมายถึงอะไร ลองพิมพ์ใหม่อีกครั้งนะคะ" if lang == "th" else "😅 I'm not sure what you meant. Could you please rephrase?"
    elif is_closing_message(user_input):
        reply = "😊 ยินดีเสมอนะคะ หากต้องการคำแนะนำเพิ่มเติมสามารถถามได้ตลอดเลยค่ะ!" if lang == "th" else "😊 You're always welcome! Feel free to ask me anything anytime!"
    else:
        with st.spinner("Thinking..."):
            corrected_input = autocorrect_text(user_input)
            question_embedding = embedding_model.encode(corrected_input).tolist()
            recommendations = retrieve_recommendations(question_embedding, top_k=10)

            prompt = f"Question: {corrected_input}\nRecommendations:\n"
            for rec in recommendations:
                prompt += f"- {rec}\n"

            prompt += """
Please generate a supportive, practical, and encouraging response based on the suggestions above.
Respond in the same language as the user's question:
- Thai if the question is in Thai.
- English if the question is in English.

Make your answer concise and natural, like a caring life coach giving motivation in just 2-3 sentences. Keep it positive and uplifting.

If replying in Thai, please use polite female ending ("ค่ะ") to sound gentle and warm.
"""

            reply = query_llm_with_chat(prompt, api_key)

    # บันทึกคำตอบ
    st.session_state.chat_history.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.markdown(reply)
