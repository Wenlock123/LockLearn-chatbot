# -*- coding: utf-8 -*-
import streamlit as st
import os
import requests
import json
import shutil

import chromadb
import together

from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document

# -- ลบ cache เดิมเพื่อความมั่นใจ
cache_dir = os.path.expanduser(
    "~/.cache/huggingface/hub/models--sentence-transformers--paraphrase-multilingual-mpnet-base-v2"
)
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)

# โหลด embedding model
embed_model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

# -- ใช้ path ที่ปลอดภัยข้ามแพลตฟอร์ม
persist_dir = os.path.join(os.path.dirname(__file__), "chromadb_database_v2")

# โหลด vector database โดยระบุชื่อ collection (ถ้าทราบ)
db = Chroma(
    persist_directory=persist_dir,
    embedding_function=HuggingFaceEmbeddings(model_name="paraphrase-multilingual-mpnet-base-v2"),
    collection_name="locklearn"  # <-- เปลี่ยนชื่อถ้า collection ของคุณไม่ใช่ "locklearn"
)

# ตั้งค่า API Key สำหรับ Together
together.api_key = os.getenv("TOGETHER_API_KEY")

PROMPT_TEMPLATE = """
You are a compassionate and wise life coach. Your goal is to respond kindly, with encouragement, and provide helpful advice in 1–3 sentences, tailored to the user's language and emotional tone.

User question:
"{question}"

Relevant background information:
{context}

Answer in the same language as the user's question (Thai or English).
"""

# ถาม LLaMA 4 Scout
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
    try:
        # สร้าง embedding
        query_embedding = embed_model.encode(user_input)

        # ดึง context
        results = db.similarity_search_by_vector([query_embedding], k=10)
        context = "\n".join([doc.page_content for doc in results])

        # เตรียม prompt
        prompt = PROMPT_TEMPLATE.format(question=user_input, context=context)

        # ส่งเข้า LLM
        with st.spinner("🧘 Thinking like a life coach..."):
            llm_response = ask_llm(prompt)

        st.markdown("### 🧭 Life Coach's Advice:")
        st.success(llm_response)

    except Exception as e:
        st.error(f"❌ Error: {e}")
