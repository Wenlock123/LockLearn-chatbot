import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
import requests

# ไม่กำหนด persist_directory เพื่อหลีกเลี่ยงปัญหา sqlite3 เวอร์ชันต่ำ
client = chromadb.Client()

embedding_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

try:
    collection = client.get_collection("recommendations")
except Exception:
    collection = client.create_collection("recommendations")

def generate_answer(question):
    question_embedding = embedding_model.encode(question).tolist()
    results = collection.query(query_embeddings=[question_embedding], n_results=10)
    context = "\n".join(results['documents'][0]) if results['documents'] else ""
    prompt = (
        f"You are a friendly and empathetic life coach. "
        f"Use the following advice to help the user with their question.\n\n"
        f"Advice:\n{context}\n\n"
        f"User question: {question}\n"
        f"Answer briefly with 1-3 sentences, encouraging and human-like."
    )
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

st.title("Life Coach Chatbot with RAG")
user_question = st.text_input("Ask your question:")

if user_question:
    with st.spinner("Thinking..."):
        answer = generate_answer(user_question)
    st.markdown("### Answer:")
    st.write(answer)
