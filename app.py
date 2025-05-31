import streamlit as st
import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import requests

# ✅ ใช้ vector store ที่อยู่ในโฟลเดอร์นี้
DB_PATH = "chromadb_database_v2"

# ✅ เรียก PersistentClient พร้อมตั้ง path ให้ถูกต้อง
client = chromadb.PersistentClient(path=DB_PATH)

# ✅ สร้าง embedding function โดยใช้โมเดล multilingual MPNet
embedding_func = SentenceTransformerEmbeddingFunction(model_name="paraphrase-multilingual-mpnet-base-v2")

# ✅ สร้างหรือโหลด collection ที่มีชื่อว่า "recommendations"
try:
    collection = client.get_collection("recommendations", embedding_function=embedding_func)
except:
    collection = client.create_collection("recommendations", embedding_function=embedding_func)

# ✅ ฟังก์ชันหลักเพื่อ generate คำตอบ
def generate_answer(question):
    question_embedding = embedding_func(question)
    
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=10,
    )
    
    context = "\n".join(results["documents"][0]) if results["documents"] else ""

    prompt = (
        f"You are a friendly and empathetic life coach. "
        f"Use the following advice to help the user with their question.\n\n"
        f"Advice:\n{context}\n\n"
        f"User question: {question}\n"
        f"Answer briefly with 1-3 sentences, encouraging and human-like."
    )

    API_URL = "https://api.together.xyz/api/v1/generate"
    API_KEY = "YOUR_API_KEY"  # 👈 เปลี่ยนเป็นของจริง
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

# ✅ UI ด้วย Streamlit
st.title("Life Coach Chatbot with RAG")
user_question = st.text_input("Ask your question:")

if user_question:
    with st.spinner("Thinking..."):
        answer = generate_answer(user_question)
    st.markdown("### Answer:")
    st.write(answer)
