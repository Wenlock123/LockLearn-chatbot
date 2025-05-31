import streamlit as st
from sentence_transformers import SentenceTransformer
import chromadb
import requests

# --- ตั้งค่า Together API สำหรับ LLaMA 4 Scout ---
TOGETHER_API_URL = "https://api.together.xyz/v0/models/meta-llama/llama-4-scout-17b-16e-instruct/generate"
TOGETHER_API_KEY = st.secrets["together_api_key"]  # แนะนำเก็บ key ใน Streamlit secrets

# --- โหลด Embedding Model ---
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

embedding_model = load_embedding_model()

# --- เชื่อมต่อกับ ChromaDB ---
@st.cache_resource
def connect_chromadb():
    db_path = './chromadb_database_v2'  # ปรับ path ให้ถูกต้อง
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_collection(name="recommendations")
    return collection

collection = connect_chromadb()

# --- ฟังก์ชันค้นหา recommendations ---
def retrieve_recommendations(question_embedding, top_k=10):
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k
    )
    if results['documents']:
        return results['documents'][0]  # รายการคำแนะนำ 10 ชิ้น
    return []

# --- ฟังก์ชันเรียก LLM เพื่อ generate response ---
def call_llm_with_context(question, retrieved_texts):
    # รวม context เป็น prompt ภาษาอังกฤษแนว Life Coach
    context = "\n".join([f"- {rec}" for rec in retrieved_texts])
    prompt = (
        f"You are a friendly and empathetic life coach. "
        f"Use the following advice to help the user with their question.\n\n"
        f"Advice:\n{context}\n\n"
        f"User question: {question}\n"
        f"Answer briefly with 1-3 sentences, encouraging and human-like."
    )
    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json"
    }
    json_data = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 256, "do_sample": True, "top_p": 0.95, "temperature": 0.7}
    }
    response = requests.post(TOGETHER_API_URL, headers=headers, json=json_data)
    response.raise_for_status()
    data = response.json()
    return data.get("results", [{}])[0].get("text", "").strip()

# --- Streamlit UI ---
st.set_page_config(page_title="LockLearn Life Coach Chatbot", page_icon="💬")
st.title("💬 LockLearn Life Coach Chatbot")

if "history" not in st.session_state:
    st.session_state.history = []

# Input text box
user_input = st.text_input("Ask me anything about life coaching:")

if user_input:
    with st.spinner("Thinking..."):
        # 1. Embed user input
        question_embedding = embedding_model.encode(user_input).tolist()

        # 2. Retrieve top 10 recommendations
        retrieved_docs = retrieve_recommendations(question_embedding, top_k=10)

        # 3. Generate answer from LLM
        answer = call_llm_with_context(user_input, retrieved_docs)

        # 4. Update chat history
        st.session_state.history.append({"user": user_input, "bot": answer})

# Display chat history like ChatGPT style
for chat in st.session_state.history:
    st.markdown(f"**You:** {chat['user']}")
    st.markdown(f"**Life Coach:** {chat['bot']}")

