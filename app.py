# -*- coding: utf-8 -*-
"""app"""

import streamlit as st
import os
import requests
import time
import json
import shutil

import chromadb
import together

from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.schema import Document

# --- ลบ cache โมเดล SentenceTransformer เก่า เพื่อแก้ปัญหาไฟล์หาย ---
cache_dir = os.path.expanduser(
    "~/.cache/huggingface/hub/models--sentence-transformers--paraphrase-multilingual-mpnet-base-v2"
)

if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)  # ลบ cache โฟลเดอร์โมเดลเก่า

# โหลด embedding model ใหม่ (จะดาวน์โหลดไฟล์โมเดลทั้งหมดใหม่)
embed_model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

# โหลด vector database จากโฟลเดอร์
db = Chroma(
    persist_directory="chromadb_database_v2",
    embedding_function=HuggingFaceEmbeddings(model_name="paraphrase-multilingual-mpnet-base-v2")
)

# ตั้งค่า API Key สำหรับ Together
together.api_key = os.getenv("TOGETHER_API_KEY")

# Prompt ภาษาอังกฤษ
PROMPT_TEMPLATE = """
You are a compassionate and wise life coach. Your goal is to respond kindly, with encouragement, and provide helpful advice in 1–3 sentences, tailored to the user's language and emotional tone.

User question:
"{question}"

Relevant background information:
{context}

Answer in the same language as the user's question (Thai or English).
"""

# ฟังก์ชันถาม LLaMA 4 Scout ผ่าน Together API
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
    # สร้าง embedding
    query_embedding = embed_model.encode(user_input)

    # 🔧 แก้ไขตรงนี้: ห่อ embedding เป็น list เพื่อให้ ChromaDB ใช้งานได้
    results = db.similarity_search_by_vector([query_embedding], k=10)

    # ตรวจสอบว่ามี context ที่เกี่ยวข้องหรือไม่
    if not results:
        st.warning("ไม่พบข้อมูลที่เกี่ยวข้อง ลองพิมพ์คำถามใหม่อีกครั้งนะครับ")
    else:
        context = "\n".join([doc.page_content for doc in results])

        # เตรียม prompt สำหรับ LLM
        prompt = PROMPT_TEMPLATE.format(question=user_input, context=context)

        # ส่งเข้า LLaMA 4 Scout
        with st.spinner("🧘 Thinking like a life coach..."):
            llm_response = ask_llm(prompt)

        st.markdown("### 🧭 Life Coach's Advice:")
        st.success(llm_response)
