import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
import requests

# กำหนดพาธเก็บฐานข้อมูล chromadb แบบถาวร
DB_PATH = "./db"

# สร้าง client chromadb พร้อมตั้งค่า persist_directory
client = chromadb.Client(
    chromadb.config.Settings(
        persist_directory=DB_PATH
    )
)

# โหลดโมเดล embedding (เช่น multilingual MPNet)
embedding_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

# สร้างหรือเปิด collection สำหรับเก็บข้อความแนะนำ (recommendations)
try:
    collection = client.get_collection("recommendations")
except Exception:
    collection = client.create_collection("recommendations")

# ฟังก์ชันค้นหา context ด้วย embedding แล้วเรียก LLM ผ่าน Together API
def generate_answer(question):
    # สร้าง embedding ของคำถาม
    question_embedding = embedding_model.encode(question).tolist()
    
    # ดึงข้อมูล top 10 ที่ใกล้เคียงที่สุด
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=10,
    )
    
    # รวม context จากผลลัพธ์
    context = "\n".join(results['documents'][0]) if results['documents'] else ""
    
    # สร้าง prompt สำหรับ LLM (Life coach)
    prompt = (
        f"You are a friendly and empathetic life coach. "
        f"Use the following advice to help the user with their question.\n\n"
        f"Advice:\n{context}\n\n"
        f"User question: {question}\n"
        f"Answer briefly with 1-3 sentences, encouraging and human-like."
    )
    
    # เรียก Together API (แก้ YOUR_API_KEY และ API_URL ให้ตรงกับของคุณ)
    API_URL = "https://api.together.xyz/api/v1/generate"
    API_KEY = "YOUR_API_KEY"
    headers = {"Authorization": f"Bearer {API_KEY}"}
    json_data = {
        "model": "meta-llama/llama-4-scout-17b-16e-instruct",
        "prompt": prompt,
        "max_new_tokens": 200,
        "temperature": 0.7,
    }
    response = requests.post(API_URL, headers=headers, json=json_data)
    if response.status_code == 200:
        return response.json().get("results", [{}])[0].get("text", "Sorry, I couldn't generate an answer.")
    else:
        return "Error contacting the language model API."

# Streamlit UI
st.title("Life Coach Chatbot with RAG")

user_question = st.text_input("Ask your question:")

if user_question:
    with st.spinner("Thinking..."):
        answer = generate_answer(user_question)
    st.markdown("### Answer:")
    st.write(answer)
