# -*- coding: utf-8 -*-
import streamlit as st
import os
import requests
import time
import json
import shutil

import chromadb
import together

from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document


# --- ลบ cache โมเดล SentenceTransformer เก่า เพื่อแก้ปัญหาไฟล์หาย ---
cache_dir = os.path.expanduser(
    "~/.cache/huggingface/hub/models--sentence-transformers--paraphrase-multilingual-mpnet-base-v2"
)

if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)

# โหลด embedding model ใหม่
embed_model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

# โหลด vector database
embedding_function = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-mpnet-base-v2")
db = Chroma(
    persist_directory="chromadb_database_v2",
    embedding_function=embedding_function
)

# ตรวจสอบว่ามีข้อมูลใน vector DB หรือไม่
try:
    doc_count = db._collection.count()
    if doc_count == 0:
        st.warning("❗ Vector database ยังไม่มีข้อมูล กรุณาเพิ่มเอกสารก่อนใช้งาน.")
except Exception as e:
    st.error(f"เกิดข้อผิดพลาดในการตรวจสอบ vector DB: {e}")
    st.stop()

# ตั้งค่า API Key
together.api_key = os.getenv("TOGETHER_API_KEY")

# Prompt Template
PROMPT_TEMPLATE = """
You are a compassionate and wise life coach. Your goal is to respond kindly, with encouragement, and provide helpful advice in 1–3 sentences, tailored to the user's language and emotional tone.

User question:
"{question}"

Relevant background information:
{context}

Answer in the same language as the user's question (Thai or English).
"""

def ask_llm(prompt):
    response = together.Complete.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        prompt=prompt,
        max_tokens=300,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
        stop=["</s>"]
    )
    return response["output"]["choices"][0]["text"].strip()

# Streamlit UI
st.title("🧠 LockLearn - Your Life Coach Chatbot")
st.markdown("Ask anything you'd like guidance or motivation about — in Thai or English!")

user_input = st.text_input("💬 What's on your mind?", placeholder="เช่น ผมรู้สึกท้อแท้...")

if user_input:
    # แก้ไขให้เป็น list of list เพื่อใช้ใน similarity_search_by_vector
    query_embedding = [embed_model.encode(user_input).tolist()]

    try:
        results = db.similarity_search_by_vector(query_embedding, k=10)
        context = "\n".join([doc.page_content for doc in results])
    except Exception as e:
        st.error(f"❌ เกิดข้อผิดพลาดในการค้นหาใน vector DB: {e}")
        st.stop()

    prompt = PROMPT_TEMPLATE.format(question=user_input, context=context)

    with st.spinner("🧘 Thinking like a life coach..."):
        llm_response = ask_llm(prompt)

    st.markdown("### 🧭 Life Coach's Advice:")
    st.success(llm_response)
