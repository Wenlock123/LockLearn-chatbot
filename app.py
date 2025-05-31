import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
import torch
import numpy as np

# ตั้งค่า Model (โหลดครั้งเดียว)
@st.cache_resource
def load_model():
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    model.eval()
    return model

# โหลด Chroma DB (In-Memory) และเตรียม Collection
@st.cache_resource
def load_chroma_collection():
    client = chromadb.Client()
    # สร้าง collection ใหม่ถ้ายังไม่มี
    try:
        collection = client.get_collection(name="recommendations")
    except:
        collection = client.create_collection(name="recommendations")
    return collection

# ฟังก์ชันแปลงข้อความเป็น embedding
def embed_text(text, model):
    embedding = model.encode([text], convert_to_tensor=True, device="cpu")
    return embedding[0].cpu().numpy()

# ฟังก์ชันดึงคำแนะนำโดยใช้ similarity search
def retrieve_recommendations(collection, question_embedding, top_k=3):
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k
    )
    # results['documents'] คือ list ของ list (batch) แต่เราใช้ batch=1
    if results['documents'] and len(results['documents'][0]) > 0:
        return results['documents'][0]
    return ["ไม่มีคำแนะนำที่ใกล้เคียง"]

# ฟังก์ชันสำหรับเก็บ log แชท (history) ไว้บน session state
def init_session_state():
    if "history" not in st.session_state:
        st.session_state.history = []

st.title("💬 LockLearn Chatbot - RAG with Chroma DB")

init_session_state()

model = load_model()
collection = load_chroma_collection()

# UI: กล่องข้อความถาม
query = st.text_input("ถามคำถามของคุณที่นี่:")

if query:
    # แปลงคำถามเป็น embedding
    query_emb = embed_text(query, model)
    
    # ดึงคำแนะนำ
    recs = retrieve_recommendations(collection, query_emb, top_k=5)

    # เก็บลง history
    st.session_state.history.append({"user": query, "bot": recs})

# แสดงประวัติแชท (user และ bot)
for chat in st.session_state.history[::-1]:  # แสดงย้อนหลังล่าสุดบน
    st.markdown(f"**คุณ:** {chat['user']}")
    st.markdown(f"**LockLearn Bot:**")
    for idx, rec in enumerate(chat['bot'], 1):
        st.markdown(f"{idx}. {rec}")
    st.markdown("---")
