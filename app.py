# ✅ TODO: If trying this app locally, comment out these 3 lines
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
import requests

# ✅ Load embedding model
@st.cache_resource
def load_embedder():
    return SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

embedder = load_embedder()

# ✅ Connect to ChromaDB
client = chromadb.PersistentClient(path="chromadb_database_v2")
collection = client.get_collection(name="recommendations")

# ✅ Function to get embedding
def get_embedding(text):
    return embedder.encode(text).tolist()

# ✅ RAG function to retrieve top K recommendations
def retrieve_recommendations(question, top_k=10):
    embedding = get_embedding(question)
    results = collection.query(query_embeddings=[embedding], n_results=top_k)
    documents = results['documents'][0] if results['documents'] else []
    return documents

# ✅ LLM function (via Together API)
def generate_answer_with_llm(question, recommendations):
    prompt = f"""You are a life coach AI. Here's a user's question: "{question}"

Based on the following recommendations:
{chr(10).join(f"- {rec}" for rec in recommendations)}

Give a personalized, encouraging, and helpful answer using the recommendations. Do not list them. Respond naturally.
"""
    response = requests.post(
        "https://api.together.xyz/inference",
        headers={
            "Authorization": "Bearer YOUR_TOGETHER_API_KEY",
            "Content-Type": "application/json"
        },
        json={
            "model": "meta-llama/llama-4-scout-17b-16e-instruct",
            "prompt": prompt,
            "max_tokens": 512,
            "temperature": 0.7,
        }
    )
    if response.status_code == 200:
        return response.json()["output"]["choices"][0]["text"].strip()
    else:
        return f"❌ Failed to get response from LLM: {response.text}"

# ✅ Streamlit UI
st.set_page_config(page_title="LockLearn AI Chatbot", page_icon="🧠")
st.title("🧠 LockLearn - Life Coaching Chatbot")

user_question = st.text_input("💬 ถามคำถามเกี่ยวกับชีวิต การเรียน หรืออนาคตของคุณ:")

if user_question:
    with st.spinner("🔍 กำลังค้นหาคำแนะนำที่เกี่ยวข้อง..."):
        recommendations = retrieve_recommendations(user_question)
    
    st.subheader("📚 คำแนะนำที่เกี่ยวข้อง")
    for i, rec in enumerate(recommendations, 1):
        st.markdown(f"{i}. {rec}")

    with st.spinner("✍️ กำลังสร้างคำตอบโดย LLM..."):
        final_answer = generate_answer_with_llm(user_question, recommendations)

    st.subheader("🤖 คำตอบจาก LockLearn AI")
    st.markdown(final_answer)
